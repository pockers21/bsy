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
  serialized_pb=b'\n\x13\x62sy/proto/bsy.proto\x12\x03\x62sy\"\x13\n\x04test\x12\x0b\n\x03num\x18\x01 \x01(\x05'
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

DESCRIPTOR.message_types_by_name['test'] = _TEST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

test = _reflection.GeneratedProtocolMessageType('test', (_message.Message,), {
  'DESCRIPTOR' : _TEST,
  '__module__' : 'bsy.proto.bsy_pb2'
  # @@protoc_insertion_point(class_scope:bsy.test)
  })
_sym_db.RegisterMessage(test)


# @@protoc_insertion_point(module_scope)