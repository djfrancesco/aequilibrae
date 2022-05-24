import sqlite3
import string
import gc
from typing import List
import importlib.util as iutil
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString

from aequilibrae.project.network.link_types import LinkTypes
from .haversine import haversine
from aequilibrae import logger
from aequilibrae.parameters import Parameters
from ...utils import WorkerThread
from ..spatialite_connection import spatialite_connection

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal

spec = iutil.find_spec("qgis")
isqgis = spec is not None
if isqgis:
    import qgis


class OSMBuilder(WorkerThread):
    if pyqt:
        building = pyqtSignal(object)

    def __init__(self, osm_items: List, path: str, node_start=10000) -> None:
        WorkerThread.__init__(self, None)
        self.osm_items = osm_items
        self.path = path
        self.conn = None
        self.node_start = node_start
        self.__link_types = None  # type: LinkTypes
        self.report = []
        self.__model_link_types = []
        self.__model_link_type_ids = []
        self.__link_type_quick_reference = {}
        self.nodes = pd.DataFrame([])
        self.links = pd.DataFrame([])
        self.insert_qry = """INSERT INTO {} ({}, geometry) VALUES({}, GeomFromWKB(?, 4326))"""

    def __emit_all(self, *args):
        if pyqt:
            self.building.emit(*args)

    def doWork(self):
        if isqgis:
            self.conn = qgis.utils.spatialite_connect(self.path)
        else:
            conn = sqlite3.connect(self.path)
            self.conn = spatialite_connection(conn)
        self.curr = self.conn.cursor()
        self.__worksetup()
        node_count = self.data_structures()
        self.importing_links(node_count)
        self.__emit_all(["finished_threaded_procedure", 0])

    def data_structures(self):
        logger.info("Separating nodes and links")
        self.__emit_all(["text", "Separating nodes and links"])
        self.__emit_all(["maxValue", len(self.osm_items)])

        alinks = []
        n = []
        tot_items = len(self.osm_items)
        # When downloading data for entire countries, memory consumption can be quite intensive
        # So we get rid of everything we don't need
        for i in range(tot_items, 0, -1):
            item = self.osm_items.pop(-1)
            if item['type'] == "way":
                alinks.append(item)
            elif item['type'] == "node":
                n.append(item)
            self.__emit_all(["Value", tot_items - i])
        gc.collect()

        logger.info("Setting data structures for nodes")
        self.__emit_all(["text", "Setting data structures for nodes"])
        self.__emit_all(["maxValue", len(n)])

        all_nodes = {}
        for i, node in enumerate(n):
            nid = node.pop("id")
            _ = node.pop("type")
            node['geo'] = Point([node['lon'], node['lat']])
            all_nodes[nid] = node

            self.__emit_all(["Value", i])

        self.nodes = pd.DataFrame(all_nodes).transpose()
        del all_nodes

        logger.info("Setting data structures for links")
        self.__emit_all(["text", "Setting data structures for links"])
        self.__emit_all(["maxValue", len(alinks)])

        all_nodes = []
        self.links = {}
        for i, link in enumerate(alinks):
            osm_id = link.pop("id")
            _ = link.pop("type")
            all_nodes.extend(link["nodes"])
            self.links[osm_id] = link
            self.__emit_all(["Value", i])
        del alinks

        logger.info("Finalizing data structures")
        self.__emit_all(["text", "Finalizing data structures"])

        return self.unique_count(np.array(all_nodes))

    def importing_links(self, node_count):

        self.__pre_process_links(node_count)
        self.__build_link_types()
        table = "links"
        fields = self.get_link_fields()
        all_fields = fields + ['geometry']
        self.__update_table_structure()
        field_names = ",".join(fields)

        missing_fields = {x: 0 for x in fields if x not in self.links.columns}
        self.links = self.links.assign(**missing_fields)

        p = Parameters()
        min_node = p.parameters["network"]["osm"]["minimum_node_id"]

        logger.info("Adding network nodes")
        nodes = self.nodes.drop_duplicates(subset=['lat', 'lon']).sort_index()
        nodes = [[min_node + i, idx, rec.geo.wkb] for i, (idx, rec) in enumerate(nodes.iterrows())]
        self.conn.executemany('insert into nodes(node_id, osm_id, geometry) values(?, ?, geomfromWKB(?,4326));', nodes)
        del nodes

        logger.info("Adding network links")
        self.__emit_all(["text", "Adding network links"])
        L = self.links.shape[0]
        self.__emit_all(["maxValue", L])

        links_sql = self.insert_qry.format(table, field_names, ','.join(['?'] * (len(fields))))
        all_link_data = []

        self.conn.commit()
        self.links = self.links[all_fields]
        self.links.direction = self.links.direction.astype(int)
        for counter, link in self.links.iterrows():
            self.__emit_all(["Value", counter])
            data = list(link.values)
            all_link_data.append(data)
            if (counter + 1) % 1000 == 0:
                self.__emit_all(["text", "Adding link chunk"])
                self.conn.executemany(links_sql, all_link_data)
                all_link_data = []
                logger.info(f'Adding links from {counter + 1:,} out of {L:,}')
                self.__emit_all(["text", "Building segments"])

        logger.info('Starting link insert')
        self.conn.executemany(links_sql, all_link_data)
        logger.info('Finished inserting links')
        self.conn.commit()

        logger.info('Cleaning node insertion')
        self.conn.execute('''Delete from Nodes where node_id not in (select a_node from links
                                                                     union all
                                                                     select b_node from links);''')
        self.conn.commit()

    def __pre_process_links(self, node_count):

        self.__update_table_structure()

        logger.info("Pre-processing links")
        self.__emit_all(["text", "Pre-processing links"])
        L = len(self.links)
        self.__emit_all(["maxValue", L])

        mode_codes, not_found_tags = self.modes_per_link_type()
        owf, twf = self.__field_osm_source()

        data = []
        for counter, (osm_id, link) in enumerate(self.links.items()):
            linktags = link["tags"]

            self.__emit_all(["Value", counter])
            if (counter + 1) % 1000 == 0:
                logger.info(f'Building segments from {counter + 1:,} out of {L:,} OSM link objects')

            vars = {"osm_id": osm_id,
                    'nodes': [link['nodes']],
                    "modes": mode_codes.get(linktags.get("highway"), not_found_tags),
                    'oneway': linktags.get("oneway", "no"),
                    'highway': linktags.get("highway")}

            if not len(vars["modes"]):
                continue

            for k, v in owf.items():
                vars[k] = linktags.get(v)

            for k, v in twf.items():
                val = linktags.get(v["osm_source"])
                if linktags.get("oneway") != "yes":
                    continue
                for d1, d2 in [("ab", "forward"), ("ba", "backward")]:
                    vars[f"{k}_{d1}"] = self.__get_link_property(d2, val, linktags, v)

            df = pd.DataFrame(vars["nodes"][0], columns=['node_id'])
            df = df.merge(node_count, how='left', on='node_id').reset_index(drop=True)
            df.fillna(value=0, inplace=True)
            df.records.values[0] += 2
            df.records.values[-1] += 2

            intersections = np.array(df[df.records >= 2].index.values)
            intersections[-1] += 1
            for i, j in zip(intersections[:-1], intersections[1:]):
                nodes = df.loc[i:j, 'node_id']
                geo = LineString(self.nodes.loc[nodes.values, 'geo'].tolist())
                vars['geometry'] = geo.wkb
                vars['distance'] = geo.length * 111000  # Approximate value, will be corrected by triggers
                data.append(pd.DataFrame(vars))
        self.__emit_all(["text", f"All {L:,} super links were pre-processed"])
        self.links = pd.concat(data).assign(direction=0)
        self.links = self.links.assign(link_id=np.arange(self.links.shape[0]) + 1, a_node=1, b_node=1)
        if 'oneway' in self.links:
            self.links.loc[self.links.oneway.str.lower() == 'yes', 'direction'] = 1

    def __build_link_types(self):
        data = []
        missing_link_types = self.links.highway.unique()
        logger.info(f'We will need to add {len(missing_link_types)} new link types to the model database')
        for link_type in missing_link_types:
            data.append([link_type, self.__repair_link_type(link_type)])
        df = pd.DataFrame(data, columns=['highway', 'link_type'])
        self.links = self.links.drop(columns=['link_type']).merge(df, on='highway')
        self.links.drop(columns=['highway', 'oneway'], inplace=True)

    def __worksetup(self):
        self.__link_types = LinkTypes(self)
        lts = self.__link_types.all_types()
        for lt_id, lt in lts.items():
            self.__model_link_types.append(lt.link_type)
            self.__model_link_type_ids.append(lt_id)

    def __update_table_structure(self):
        curr = self.conn.cursor()
        curr.execute('pragma table_info(Links)')
        structure = curr.fetchall()
        has_fields = [x[1].lower() for x in structure]
        fields = [field.lower() for field in self.get_link_fields()] + ['osm_id']
        for field in [f for f in fields if f not in has_fields]:
            ltype = self.get_link_field_type(field).upper()
            curr.execute(f'Alter table Links add column {field} {ltype}')
        self.conn.commit()

    def __build_link_data(self, vars, intersections, i, linknodes, node_ids, fields):
        ii = intersections[i]
        jj = intersections[i + 1]
        all_nodes = [linknodes[x] for x in range(ii, jj + 1)]

        vars["a_node"] = node_ids.get(linknodes[ii], self.node_start)
        if vars["a_node"] == self.node_start:
            node_ids[linknodes[ii]] = vars["a_node"]
            self.node_start += 1

        vars["b_node"] = node_ids.get(linknodes[jj], self.node_start)
        if vars["b_node"] == self.node_start:
            node_ids[linknodes[jj]] = vars["b_node"]
            self.node_start += 1

        vars["distance"] = sum([haversine(self.nodes[x]["lon"], self.nodes[x]["lat"],
                                          self.nodes[y]["lon"], self.nodes[y]["lat"])
                                for x, y in zip(all_nodes[1:], all_nodes[:-1])])

        geometry = ["{} {}".format(self.nodes[x]["lon"], self.nodes[x]["lat"]) for x in all_nodes]
        geometry = "LINESTRING ({})".format(", ".join(geometry))

        attributes = [vars.get(x) for x in fields]
        attributes.append(geometry)
        return attributes

    def __repair_link_type(self, link_type: str) -> str:
        original_link_type = link_type
        link_type = ''.join([x for x in link_type if x in string.ascii_letters + '_']).lower()

        split = link_type.split('_')
        for i, piece in enumerate(split[1:]):
            if piece in ['link', 'segment', 'stretch']:
                link_type = '_'.join(split[0:i + 1])

        if len(link_type) == 0:
            link_type = 'empty'

        if len(self.__model_link_type_ids) >= 51 and link_type not in self.__model_link_types:
            link_type = 'aggregate_link_type'

        if link_type in self.__model_link_types:
            lt = self.__link_types.get_by_name(link_type)
            if original_link_type not in lt.description:
                lt.description += f', {original_link_type}'
                lt.save()
            self.__link_type_quick_reference[original_link_type.lower()] = link_type
            return link_type

        letter = link_type[0]
        if letter in self.__model_link_type_ids:
            letter = letter.upper()
            if letter in self.__model_link_type_ids:
                for letter in string.ascii_letters:
                    if letter not in self.__model_link_type_ids:
                        break
        lt = self.__link_types.new(letter, False)
        lt.link_type = link_type
        lt.description = f"Link types from Open Street Maps: {original_link_type}"
        lt.save()
        self.__model_link_types.append(link_type)
        self.__model_link_type_ids.append(letter)
        self.__link_type_quick_reference[original_link_type.lower()] = link_type
        return link_type

    def __get_link_property(self, d2, val, linktags, v):
        vald = linktags.get(f"{v['osm_source']}:{d2}", val)
        if vald is None:
            return vald

        if vald.isdigit():
            if vald == val and v["osm_behaviour"] == "divide":
                vald = float(val) / 2
        return vald

    @staticmethod
    def unique_count(a):
        # From: https://stackoverflow.com/a/21124789/1480643
        unique, inverse = np.unique(a, return_inverse=True)
        count = np.zeros(len(unique), int)
        np.add.at(count, inverse, 1)
        df = pd.DataFrame(np.vstack((unique, count)).T, columns=['node_id', 'records'])
        df = df[df.records > 1]
        return df

    @staticmethod
    def get_link_fields():
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]
        owf = [list(x.keys())[0] for x in fields["one-way"]]

        twf1 = ["{}_ab".format(list(x.keys())[0]) for x in fields["two-way"]]
        twf2 = ["{}_ba".format(list(x.keys())[0]) for x in fields["two-way"]]

        return owf + twf1 + twf2 + ["osm_id"]

    @staticmethod
    def get_link_field_type(field_name):
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        if field_name[-3:].lower() in ['_ab', '_ba']:
            field_name = field_name[:-3]
            for tp in fields["two-way"]:
                if field_name in tp:
                    return tp[field_name]['type']
        else:

            for tp in fields["one-way"]:
                if field_name in tp:
                    return tp[field_name]['type']

    @staticmethod
    def __field_osm_source():
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]

        owf = {
            list(x.keys())[0]: x[list(x.keys())[0]]["osm_source"]
            for x in fields["one-way"]
            if "osm_source" in x[list(x.keys())[0]]
        }

        twf = {}
        for x in fields["two-way"]:
            if "osm_source" in x[list(x.keys())[0]]:
                twf[list(x.keys())[0]] = {
                    "osm_source": x[list(x.keys())[0]]["osm_source"],
                    "osm_behaviour": x[list(x.keys())[0]]["osm_behaviour"],
                }
        return owf, twf

    def modes_per_link_type(self):
        p = Parameters()
        modes = p.parameters["network"]["osm"]["modes"]

        cursor = self.conn.cursor()
        cursor.execute("SELECT mode_name, mode_id from modes")
        mode_codes = cursor.fetchall()
        mode_codes = {p[0]: p[1] for p in mode_codes}

        type_list = {}
        notfound = ""
        for mode, val in modes.items():
            all_types = val["link_types"]
            md = mode_codes[mode]
            for tp in all_types:
                type_list[tp] = "{}{}".format(type_list.get(tp, ""), md)
            if val["unknown_tags"]:
                notfound += md

        type_list = {k: "".join(set(v)) for k, v in type_list.items()}

        return type_list, '{}'.format(notfound)

    @staticmethod
    def get_node_fields():
        p = Parameters()
        fields = p.parameters["network"]["nodes"]["fields"]
        fields = [list(x.keys())[0] for x in fields]
        return fields + ["osm_id"]
