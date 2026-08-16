"""Client adapter metadata and composition interfaces."""

from .bank import AdapterBank, AdapterBankError, AdapterDescriptor

__all__ = ["AdapterBank", "AdapterBankError", "AdapterDescriptor"]

