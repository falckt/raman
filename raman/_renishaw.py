# Author: Tillmann Falck <tf-raman@lucidus.de>
#
# License: BSD 3 clause
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, BinaryIO, Dict, Union

import warnings
import struct
import pathlib

import numpy as np
from . import _renishaw_types as rw_types

def wn_to_wl(x: float) -> float:
    return 1 / (x * 1e2) / 1e-9

def parse_wdf1(
        data: bytes,
        parsed_data: Dict[str, Any]  # pylint: disable=unused-argument
        ) -> Dict[str, Any]:
    fields = [
        ('padding1', '44s'),
        ('points_per_spectrum', 'L'),
        ('target_num_spectra', 'Q'),
        ('measured_spectra', 'Q'),
        ('accumulation_count', 'L'),
        ('ylist_length', 'L'),
        ('xlist_length', 'L'),
        ('data_origin_count', 'L'),
        ('application_name', '24s'),
        ('application_version', '8s'),
        ('scan_type', 'L'),
        ('measurement_type', 'L'),
        ('padding2', '16s'),
        ('spectral_units', 'L'),
        ('laser_wavenumber', 'f'),
        ('padding3', '48s'),
        ('username', '32s'),
        ('title', '272s')
    ]
    utf = [
        'application_name',
        'username',
        'title',
    ]
    convert_type = {
        'scan_type': rw_types.ScanType,
        'measurement_type': rw_types.MeasurementType,
        'spectral_units': rw_types.UnitType,
    }

    res = struct.unpack('<' + ''.join(f for _, f in fields), data)
    infos = dict(zip((n for n, _ in fields), res))

    for f in utf:
        infos[f] = infos[f].decode('utf8').rstrip('\x00')

    for f, t in convert_type.items():
        infos[f] = t(infos[f]).name

    infos['application_version'] = struct.unpack('<4H', infos['application_version'])
    infos['laser_wavelength'] = wn_to_wl(infos['laser_wavenumber'])

    return infos

def parse_lst(
        data: bytes,
        parsed_data: Dict[str, Any],
        length_field: str
        ):
    N = parsed_data['WDF1'][length_field]

    type_, units = struct.unpack('<LL', data[:8])

    if 4 * N != (len(data) - 8):
        raise ValueError('Size given by metadata does not agree with data block length')

    data = np.frombuffer(data[8:], np.float32, count=N)

    return {
        'type': rw_types.DataType(type_).name,
        'units': rw_types.UnitType(units).name,
        'data': data,
    }

def parse_data(data: bytes, parsed_data: Dict[str, Any]):
    M = parsed_data['WDF1']['points_per_spectrum']
    N = parsed_data['WDF1']['measured_spectra']

    return np.frombuffer(data, dtype=np.float32).reshape((N, M))

def parse_wmap(
        data: bytes,
        parsed_data: Dict[str, Any]  # pylint: disable=unused-argument
        ):
    fields = [
        ('padding', '8s'),
        ('x0', 'f'),
        ('y0', 'f'),
        ('unkwn1', 'f'),
        ('x_incr', 'f'),
        ('y_incr', 'f'),
        ('unkwn2', 'f'),
        ('width', 'L'),
        ('height', 'L'),
        ('unkwn3', '8s'),
    ]

    fmt = '<' + ''.join(f for _, f in fields)
    field_names = [n for n, _ in fields]

    unpacked = struct.unpack(fmt, data)
    infos = dict(zip(field_names, unpacked))

    return infos

def parse_origin(
        data: bytes,
        parsed_data: Dict[str, Any]  # pylint: disable=unused-argument
        ):
    N = parsed_data['WDF1']['measured_spectra']
    D = parsed_data['WDF1']['data_origin_count']

    D0, padding = struct.unpack('<b3s', data[:4])  # pylint: disable=unused-variable

    fields = [
        ('type', 'H'),
        ('xy_indicator', 'H'),
        ('units', 'L'),
        ('name', '16s'),
        ('data', f'{8*N}s'),
    ]

    fmt = struct.Struct('<' + ''.join(f for _, f in fields))
    field_names = [n for n, _ in fields]

    info_blocks = {}
    for unpacked in fmt.iter_unpack(data[4:]):
        block = dict(zip(field_names, unpacked))
        block['data'] = np.frombuffer(block['data'], np.float64)

        name = block['name'] = block['name'].decode('utf8').rstrip('\x00')
        block['units'] = rw_types.UnitType(block['units']).name
        block['type'] = rw_types.DataType(block['type']).name

        info_blocks[name] = block

    if D != len(info_blocks) or D != D0:
        raise ValueError('Something is wrong with the metadata')

    return info_blocks

def parse_text(
        data: bytes,
        parsed_data: Dict[str, Any]  # pylint: disable=unused-argument
        ):
    return data.decode('utf8').rstrip('\00')

PARSERS = {
    'WDF1': parse_wdf1,
    'XLST': lambda x, y: parse_lst(x, y, 'xlist_length'),
    'YLST': lambda x, y: parse_lst(x, y, 'ylist_length'),
    'DATA': parse_data,
    'ORGN': parse_origin,
    'TEXT': parse_text,
    'WMAP': parse_wmap,
}

# The fields
#  - WXDA
#  - WXDM
#  - ZLDC
#  - WXCS
#  - WXIS
# all seem to have similar structure and contain metadata

def read_block(fid: BinaryIO):
    data = fid.read(16)

    if data:
        block_name, block_type, block_len = struct.unpack('<4sLQ', data)  # pylint: disable=unused-variable
        block_data = fid.read(block_len - 16)

        return block_name.decode('ascii'), block_data
    else:
        return False

def consistency_checks(blocks: Dict[str, Any]):
    N = blocks['WDF1']['measured_spectra']
    W = blocks['WMAP']['width']
    H = blocks['WMAP']['height']

    if N != W*H:
        raise ValueError('Dimension mismatch')

    x_data = blocks['ORGN']['X']['data'].reshape((H, W))
    x0, x_incr = blocks['WMAP']['x0'], blocks['WMAP']['x_incr']

    if not np.allclose(x_data[0, 0], x0):
        warnings.warn('Nominal initial x position deviates from actual one')

    if not np.allclose(np.diff(x_data), x_incr):
        warnings.warn('Nominal x position increment deviate from actual ones')

    y_data = blocks['ORGN']['Y']['data'].reshape((H, W))
    y0, y_incr = blocks['WMAP']['y0'], blocks['WMAP']['y_incr']

    if not np.allclose(y_data[0, 0], y0):
        warnings.warn('Nominal initial y position deviates from actual one')

    if not np.allclose(np.diff(y_data, axis=0), y_incr):
        warnings.warn('Nominal y position increment deviate from actual ones')

def parse_wdf(path: Union[str, pathlib.Path]):
    blocks: Dict[str, Any] = {}

    with open(path, 'rb') as fid:
        while block := read_block(fid):
            name, data = block

            if name in PARSERS:
                blocks[name] = PARSERS[name](data, blocks)

    consistency_checks(blocks)

    return blocks
