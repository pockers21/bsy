# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: bsy/proto/bsy.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='bsy/proto/bsy.proto',
  package='bsy',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x13\x62sy/proto/bsy.proto\x12\x03\x62sy\"\x13\n\x04test\x12\x0b\n\x03num\x18\x01 \x01(\x05\"!\n\x0e\x44\x61taBlockShape\x12\x0f\n\x03\x64im\x18\x01 \x03(\x03\x42\x02\x10\x01\"\x8a\x01\n\x0e\x44\x61taBlockProto\x12\"\n\x05shape\x18\x01 \x01(\x0b\x32\x13.bsy.DataBlockShape\x12\x10\n\x04\x64\x61ta\x18\x02 \x03(\x02\x42\x02\x10\x01\x12\x10\n\x04\x64iff\x18\x03 \x03(\x02\x42\x02\x10\x01\x12\x17\n\x0b\x64ouble_data\x18\x04 \x03(\x01\x42\x02\x10\x01\x12\x17\n\x0b\x64ouble_diff\x18\x05 \x03(\x01\x42\x02\x10\x01\"\x90\x01\n\x1c\x44istributeGeneratorParameter\x12\x16\n\x04type\x18\x01 \x01(\t:\x08\x63onstant\x12\x13\n\x08\x63onstant\x18\x02 \x01(\x02:\x01\x30\x12\x0e\n\x03max\x18\x04 \x01(\x02:\x01\x31\x12\x0f\n\x04mean\x18\x05 \x01(\x02:\x01\x30\x12\x0e\n\x03std\x18\x06 \x01(\x02:\x01\x31\x12\x12\n\x06sparse\x18\x07 \x01(\x05:\x02-1'
)




_TEST = _descriptor.Descriptor(
  name='test',
  full_name='bsy.test',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num', full_name='bsy.test.num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=47,
)


_DATABLOCKSHAPE = _descriptor.Descriptor(
  name='DataBlockShape',
  full_name='bsy.DataBlockShape',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dim', full_name='bsy.DataBlockShape.dim', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=82,
)


_DATABLOCKPROTO = _descriptor.Descriptor(
  name='DataBlockProto',
  full_name='bsy.DataBlockProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='bsy.DataBlockProto.shape', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data', full_name='bsy.DataBlockProto.data', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='diff', full_name='bsy.DataBlockProto.diff', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='double_data', full_name='bsy.DataBlockProto.double_data', index=3,
      number=4, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='double_diff', full_name='bsy.DataBlockProto.double_diff', index=4,
      number=5, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=b'\020\001', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=223,
)


_DISTRIBUTEGENERATORPARAMETER = _descriptor.Descriptor(
  name='DistributeGeneratorParameter',
  full_name='bsy.DistributeGeneratorParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='bsy.DistributeGeneratorParameter.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=b"constant".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='constant', full_name='bsy.DistributeGeneratorParameter.constant', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='max', full_name='bsy.DistributeGeneratorParameter.max', index=2,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mean', full_name='bsy.DistributeGeneratorParameter.mean', index=3,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='std', full_name='bsy.DistributeGeneratorParameter.std', index=4,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sparse', full_name='bsy.DistributeGeneratorParameter.sparse', index=5,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=-1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=226,
  serialized_end=370,
)

_DATABLOCKPROTO.fields_by_name['shape'].message_type = _DATABLOCKSHAPE
DESCRIPTOR.message_types_by_name['test'] = _TEST
DESCRIPTOR.message_types_by_name['DataBlockShape'] = _DATABLOCKSHAPE
DESCRIPTOR.message_types_by_name['DataBlockProto'] = _DATABLOCKPROTO
DESCRIPTOR.message_types_by_name['DistributeGeneratorParameter'] = _DISTRIBUTEGENERATORPARAMETER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

test = _reflection.GeneratedProtocolMessageType('test', (_message.Message,), {
  'DESCRIPTOR' : _TEST,
  '__module__' : 'bsy.proto.bsy_pb2'
  # @@protoc_insertion_point(class_scope:bsy.test)
  })
_sym_db.RegisterMessage(test)

DataBlockShape = _reflection.GeneratedProtocolMessageType('DataBlockShape', (_message.Message,), {
  'DESCRIPTOR' : _DATABLOCKSHAPE,
  '__module__' : 'bsy.proto.bsy_pb2'
  # @@protoc_insertion_point(class_scope:bsy.DataBlockShape)
  })
_sym_db.RegisterMessage(DataBlockShape)

DataBlockProto = _reflection.GeneratedProtocolMessageType('DataBlockProto', (_message.Message,), {
  'DESCRIPTOR' : _DATABLOCKPROTO,
  '__module__' : 'bsy.proto.bsy_pb2'
  # @@protoc_insertion_point(class_scope:bsy.DataBlockProto)
  })
_sym_db.RegisterMessage(DataBlockProto)

DistributeGeneratorParameter = _reflection.GeneratedProtocolMessageType('DistributeGeneratorParameter', (_message.Message,), {
  'DESCRIPTOR' : _DISTRIBUTEGENERATORPARAMETER,
  '__module__' : 'bsy.proto.bsy_pb2'
  # @@protoc_insertion_point(class_scope:bsy.DistributeGeneratorParameter)
  })
_sym_db.RegisterMessage(DistributeGeneratorParameter)


_DATABLOCKSHAPE.fields_by_name['dim']._options = None
_DATABLOCKPROTO.fields_by_name['data']._options = None
_DATABLOCKPROTO.fields_by_name['diff']._options = None
_DATABLOCKPROTO.fields_by_name['double_data']._options = None
_DATABLOCKPROTO.fields_by_name['double_diff']._options = None
# @@protoc_insertion_point(module_scope)
