"""NVSwitch multicast memory for the multi-GPU cluster forward (hybrid_fwc.cu and hybrid_fwc_tmem.cu with FRNN_P > 1; see
docs/flashrnn2/tmem-beta-handoff.md, "Multi-GPU").

A MulticastGroup owns one CUDA multicast object over P devices of one process (driver API through ctypes): every
device binds an equally sized physical allocation; a store to the multicast virtual address is replicated by the
NVSwitch into every device's copy; each device reads its own copy through a unicast mapping. The unicast mappings
are made accessible to every device of the group, so one device can also write a peer's copy directly (the unicast
fan-out that the multicast replaces; the probe compares the two).

Requirements: devices in one NVLink/NVSwitch domain with CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, driver >= 535,
one process. Untested until it runs on such a machine (no NVSwitch machine was reachable when this was written).
"""
import ctypes
import os

import torch

CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MULTICAST_GRANULARITY_MINIMUM = 0
CU_MULTICAST_GRANULARITY_RECOMMENDED = 1
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", CUmemAllocFlags)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class CUmulticastObjectProp(ctypes.Structure):
    _fields_ = [("numDevices", ctypes.c_uint), ("size", ctypes.c_size_t), ("handleTypes", ctypes.c_ulonglong),
                ("flags", ctypes.c_ulonglong)]


_LIB = None


def driver():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(os.environ.get("FRNN_LIBCUDA", "libcuda.so.1"))
        lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        _LIB = lib
    return _LIB


def check(rc, what):
    if rc != 0:
        msg = ctypes.c_char_p()
        driver().cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{what}: CUDA driver error {rc} ({(msg.value or b'?').decode()})")


def multicast_supported(device_index):
    lib = driver()
    check(lib.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(lib.cuDeviceGet(ctypes.byref(dev), device_index), "cuDeviceGet")
    v = ctypes.c_int()
    check(lib.cuDeviceGetAttribute(ctypes.byref(v), CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev), "cuDeviceGetAttribute")
    return bool(v.value)


def _roundup(x, g):
    return (x + g - 1) // g * g


class MulticastGroup:
    """One multicast object over `devices` (CUDA ordinals), `nbytes` usable bytes (rounded up to the granularity).

    Attributes: size (bytes actually allocated per device), mc_ptr (multicast address, the same for every device),
    uc_ptr[i] (device i's unicast copy, accessible from every device of the group)."""

    def __init__(self, devices, nbytes):
        self.devices = [int(d) for d in devices]
        lib = driver()
        check(lib.cuInit(0), "cuInit")
        for d in self.devices:  # the primary contexts exist (torch creates them lazily)
            torch.zeros(1, device=f"cuda:{d}")
            if not multicast_supported(d):
                raise RuntimeError(f"cuda:{d} reports no multicast support")
        self.cudevs = []
        for d in self.devices:
            cd = ctypes.c_int()
            check(lib.cuDeviceGet(ctypes.byref(cd), d), "cuDeviceGet")
            self.cudevs.append(cd)
        prop = CUmulticastObjectProp(numDevices=len(self.devices), size=nbytes,
                                     handleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, flags=0)
        gran = ctypes.c_size_t()
        check(lib.cuMulticastGetGranularity(ctypes.byref(gran), ctypes.byref(prop), CU_MULTICAST_GRANULARITY_RECOMMENDED),
              "cuMulticastGetGranularity")
        self.granularity = gran.value
        self.size = _roundup(nbytes, self.granularity)
        prop.size = self.size
        self.mc_handle = ctypes.c_ulonglong()
        check(lib.cuMulticastCreate(ctypes.byref(self.mc_handle), ctypes.byref(prop)), "cuMulticastCreate")
        for cd in self.cudevs:
            check(lib.cuMulticastAddDevice(self.mc_handle, cd), "cuMulticastAddDevice")
        # Physical memory per device, bound into the multicast object at offset 0.
        self.uc_handles = []
        for d in self.devices:
            ap = CUmemAllocationProp(type=CU_MEM_ALLOCATION_TYPE_PINNED,
                                     requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                                     location=CUmemLocation(type=CU_MEM_LOCATION_TYPE_DEVICE, id=d))
            g2 = ctypes.c_size_t()
            check(lib.cuMemGetAllocationGranularity(ctypes.byref(g2), ctypes.byref(ap), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                  "cuMemGetAllocationGranularity")
            if self.size % g2.value:
                raise RuntimeError(f"multicast granularity {self.granularity} is not a multiple of the allocation "
                                   f"granularity {g2.value}")
            h = ctypes.c_ulonglong()
            check(lib.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(self.size), ctypes.byref(ap), ctypes.c_ulonglong(0)),
                  "cuMemCreate")
            self.uc_handles.append(h)
        for h in self.uc_handles:
            check(lib.cuMulticastBindMem(self.mc_handle, ctypes.c_size_t(0), h, ctypes.c_size_t(0),
                                         ctypes.c_size_t(self.size), ctypes.c_ulonglong(0)), "cuMulticastBindMem")
        access = (CUmemAccessDesc * len(self.devices))()
        for i, d in enumerate(self.devices):
            access[i].location = CUmemLocation(type=CU_MEM_LOCATION_TYPE_DEVICE, id=d)
            access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        # Multicast mapping (stores from any device of the group replicate).
        self.mc_ptr = ctypes.c_ulonglong()
        check(lib.cuMemAddressReserve(ctypes.byref(self.mc_ptr), ctypes.c_size_t(self.size),
                                      ctypes.c_size_t(self.granularity), ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)),
              "cuMemAddressReserve(mc)")
        check(lib.cuMemMap(self.mc_ptr, ctypes.c_size_t(self.size), ctypes.c_size_t(0), self.mc_handle,
                           ctypes.c_ulonglong(0)), "cuMemMap(mc)")
        check(lib.cuMemSetAccess(self.mc_ptr, ctypes.c_size_t(self.size), access, ctypes.c_size_t(len(self.devices))),
              "cuMemSetAccess(mc)")
        # Unicast mappings: device i's copy, readable and writable by every device of the group.
        self.uc_ptr = []
        for h in self.uc_handles:
            p = ctypes.c_ulonglong()
            check(lib.cuMemAddressReserve(ctypes.byref(p), ctypes.c_size_t(self.size), ctypes.c_size_t(self.granularity),
                                          ctypes.c_ulonglong(0), ctypes.c_ulonglong(0)), "cuMemAddressReserve(uc)")
            check(lib.cuMemMap(p, ctypes.c_size_t(self.size), ctypes.c_size_t(0), h, ctypes.c_ulonglong(0)), "cuMemMap(uc)")
            check(lib.cuMemSetAccess(p, ctypes.c_size_t(self.size), access, ctypes.c_size_t(len(self.devices))),
                  "cuMemSetAccess(uc)")
            self.uc_ptr.append(p)
        self.zero()

    def zero(self):
        """Zero every device's copy (synchronously, through the unicast mappings). Driver calls that need a current
        context run under the device's primary context, pushed and popped here: the host thread has none bound by
        default (torch's lazy init leaves none; B300 run 1 failed in cuMemsetD8 with error 201 otherwise)."""
        lib = driver()
        for cd, p in zip(self.cudevs, self.uc_ptr):
            ctx = ctypes.c_void_p()
            check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), cd), "cuDevicePrimaryCtxRetain")
            check(lib.cuCtxPushCurrent(ctx), "cuCtxPushCurrent")
            try:
                check(lib.cuMemsetD8(p, ctypes.c_ubyte(0), ctypes.c_size_t(self.size)), "cuMemsetD8")
                check(lib.cuCtxSynchronize(), "cuCtxSynchronize")
            finally:
                old = ctypes.c_void_p()
                lib.cuCtxPopCurrent(ctypes.byref(old))
                lib.cuDevicePrimaryCtxRelease(cd)

    def mc(self, offset=0):
        return self.mc_ptr.value + offset

    def uc(self, i, offset=0):
        return self.uc_ptr[i].value + offset

    def close(self):
        lib = driver()
        for p in getattr(self, "uc_ptr", []):
            lib.cuMemUnmap(p, ctypes.c_size_t(self.size))
            lib.cuMemAddressFree(p, ctypes.c_size_t(self.size))
        if getattr(self, "mc_ptr", None) is not None:
            lib.cuMemUnmap(self.mc_ptr, ctypes.c_size_t(self.size))
            lib.cuMemAddressFree(self.mc_ptr, ctypes.c_size_t(self.size))
        for cd in getattr(self, "cudevs", []):
            lib.cuMulticastUnbind(self.mc_handle, cd, ctypes.c_size_t(0), ctypes.c_size_t(self.size))
        for h in getattr(self, "uc_handles", []):
            lib.cuMemRelease(h)
        if getattr(self, "mc_handle", None) is not None:
            lib.cuMemRelease(self.mc_handle)
        self.uc_ptr, self.mc_ptr, self.uc_handles, self.mc_handle, self.cudevs = [], None, [], None, []


_GROUPS = {}


def group(devices, nbytes):
    """A cached MulticastGroup for (devices, nbytes); groups live until the process exits."""
    key = (tuple(int(d) for d in devices), int(nbytes))
    if key not in _GROUPS:
        _GROUPS[key] = MulticastGroup(key[0], key[1])
    return _GROUPS[key]
