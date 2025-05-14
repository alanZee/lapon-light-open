# LaPON, GPL-3.0 license
# Utility functions for xarray datasets
# https://github.com/google/jax-cfd


import xarray
import functools
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import numpy as np
import pandas

# names of coordinates and attributes in xarray `Dataset`.

XR_VELOCITY_NAMES = ('u', 'v', 'w')
XR_SCALAR_NAMES = ('c')
XR_SPATIAL_DIMS = ('x', 'y', 'z')
XR_WAVENUMBER_DIMS = ('kx', 'ky', 'kz')
XR_SAMPLE_NAME = 'sample'
XR_TIME_NAME = 'time'
XR_OFFSET_NAME = 'offset'

XR_SAVE_GRID_SIZE_ATTR_NAME = 'save_grid_size'
XR_SAVE_GRID_SIZE_ATTR_NAME_RECTANGLE = ('save_grid_size_x', 'save_grid_size_y')

XR_DOMAIN_SIZE_NAME = 'domain_size'
XR_NDIM_ATTR_NAME = 'ndim'
XR_STABLE_TIME_STEP_ATTR_NAME = 'stable_time_step'


def g_to_normal_(ds: xarray.Dataset) -> xarray.Dataset:
    # trans GOOGLE jax-cfd dataset to AI normal style (u/v['sample', 'time', 'x', 'y'] -> ['sample', 'time', 'channel', 'x'(h), 'y'(w)])
    u_ = ds.u.values[:, :, None, ...]
    v_ = ds.v.values[:, :, None, ...]
    velocity = np.concatenate([u_, v_], axis=2)
    
    velocity = xarray.DataArray(velocity, dims=['sample', 'time', 'channel', 'x', 'y'])
    ds_new = xarray.Dataset({
        'velocity': velocity,
        })
    for k in ds.coords.keys():
        ds_new.coords[k] = ds.coords[k]
    ds_new.coords['channel'] = ['u', 'v']
    ds_new.attrs = ds.attrs
    ds_new.attrs.update(dict(
        dx = (ds_new.x[1] - ds_new.x[0]).item(),
        dt = (ds_new.time[1] - ds_new.time[0]).item(),
        )) 
    
    return ds_new


def resize_64_to_32(ds: xarray.Dataset) -> xarray.Dataset:
    # https://github.com/google/jax-cfd/blob/main/notebooks/ml_accelerated_cfd_data_analysis.ipynb
    coarse = xarray.Dataset({
        'u': ds.u.isel(x=slice(1, None, 2)).coarsen(y=2, coord_func='max').mean(),
        'v': ds.v.isel(y=slice(1, None, 2)).coarsen(x=2, coord_func='max').mean(),
    })
    coarse.attrs = ds.attrs
    '''
    ds.u.isel(x=slice(1, None, 2)).coarsen(y=2, coord_func='max').mean()
        从 ds 中选择 u 数据数组。
        在 x 维度上，从第二个点开始每隔一个点选择数据。
        在 y 维度上，将数据点进行分组，相邻的为一组，每组 2 个数据，每组的坐标设定为该组内所有数据点坐标中的最大值。
        最后，在上一步分的每个组中用 取平均值 来完成降采样。
    '''
    return coarse


def resize_reduce_n_(ds: xarray.Dataset, n: int = 2) -> xarray.Dataset:
    # Resize(reduced) frame size (Reduced by a factor of n), i.e. n=2: 64to32, 1024to512, ...
    def resize_func(array: xarray.DataArray, n: int):
        # u = array.sel(channel='u').isel(x=slice(n-1, None, n)).coarsen(y=n, coord_func='max').mean()
        # v = array.sel(channel='v').isel(y=slice(n-1, None, n)).coarsen(x=n, coord_func='max').mean()
        u = array.sel(channel='u').isel(x=slice(n-1, None, n)).isel(y=slice(n-1, None, n))
        v = array.sel(channel='v').isel(y=slice(n-1, None, n)).isel(x=slice(n-1, None, n))
        
        u_ = u.values[:, :, None, ...]
        u_ = xarray.DataArray(u_, dims=['sample', 'time', 'channel', 'x', 'y'])
        for k in u.coords.keys():
            if k != 'channel':
                u_.coords[k] = u.coords[k].values
        u_.coords['channel'] = ['u']
        
        v_ = v.values[:, :, None, ...]
        v_ = xarray.DataArray(v_, dims=['sample', 'time', 'channel', 'x', 'y'])
        for k in v.coords.keys():
            if k != 'channel':
                v_.coords[k] = v.coords[k].values
        v_.coords['channel'] = ['v']
        
        return xarray.concat([u_, v_], dim='channel')
    
    if n == 1:
      return ds
    elif n < 1:
      raise ValueError(f'n(={n}) must be >= 1')
    else:
      coarse = ds.apply(resize_func, n=n)
      coarse.attrs = ds.attrs
      coarse.attrs['grid_size'] = coarse.dims['x'] # shape[-1]
      coarse.attrs['dx'] = (coarse.x[1] - coarse.x[0]).item()
      return coarse


def vorticity_2d(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate vorticity on a 2D dataset."""
  # Vorticity is calculated from staggered velocities at offset=(1, 1).
  dy = ds.y[1] - ds.y[0]
  dx = ds.x[1] - ds.x[0]
  dv_dx = (ds.v.roll(x=-1, roll_coords=False) - ds.v) / dx
  du_dy = (ds.u.roll(y=-1, roll_coords=False) - ds.u) / dy
  return (dv_dx - du_dy).rename('vorticity')


def vorticity_2d_(ds: xarray.Dataset, key: str = 'prediction_velocity') -> xarray.DataArray:
  """Calculate vorticity on a 2D dataset."""
  # Vorticity is calculated from staggered velocities at offset=(1, 1).
  dy = ds.y[1] - ds.y[0]
  dx = ds.x[1] - ds.x[0]
  dv_dx = (ds[key].sel(channel='v').roll(x=-1, roll_coords=False) - ds[key].sel(channel='v')) / dx
  du_dy = (ds[key].sel(channel='u').roll(y=-1, roll_coords=False) - ds[key].sel(channel='u')) / dy
  return (dv_dx - du_dy).rename('vorticity')


def enstrophy_2d(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate entrosphy over a 2D dataset."""
  return (vorticity_2d(ds) ** 2 / 2).rename('enstrophy')


def magnitude(
    u: xarray.DataArray,
    v: Optional[xarray.DataArray] = None,
    w: Optional[xarray.DataArray] = None,
) -> xarray.DataArray:
  """Calculate the magnitude of a velocity field."""
  total = sum((c * c.conj()).real for c in [u, v, w] if c is not None)
  return total ** 0.5


def speed(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate speed at each point in a velocity field."""
  args = [ds[k] for k in XR_VELOCITY_NAMES if k in ds]
  return magnitude(*args).rename('speed')


def kinetic_energy(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate kinetic energy at each point in a velocity field."""
  return (speed(ds) ** 2 / 2).rename('kinetic_energy')


def fourier_transform(array: xarray.DataArray) -> xarray.DataArray:
  """Calculate the fourier transform of an array, with labeled coordinates."""
  dims = [dim for dim in XR_SPATIAL_DIMS if dim in array.dims]
  axes = [-1, -2, -3][:len(dims)]
  result = xarray.apply_ufunc(
      functools.partial(np.fft.fftn, axes=axes), array,
      input_core_dims=[dims],
      output_core_dims=[['k' + d for d in dims]],
      output_sizes={'k' + d: array.sizes[d] for d in dims},
      output_dtypes=[np.complex128],
      dask='parallelized')
  for d in dims:
    step = float(array.coords[d][1] - array.coords[d][0])
    freqs = 2 * np.pi * np.fft.fftfreq(array.sizes[d], step)
    result.coords['k' + d] = freqs
  # Ensure frequencies are in ascending order (equivalent to fftshift)
  rolls = {'k' + d: array.sizes[d] // 2 for d in dims}
  return result.roll(rolls, roll_coords=True)


def periodic_correlate(u, v):
  """Periodic correlation of arrays `u`, `v`, implemented using the FFT."""
  return np.fft.ifft(np.fft.fft(u).conj() * np.fft.fft(v)).real


def spatial_autocorrelation(array, spatial_axis='x'):
  """Computes spatial autocorrelation of `array` along `spatial_axis`."""
  spatial_axis_size = array.sizes[spatial_axis]
  out_axis_name = 'd' + spatial_axis
  full_result = xarray.apply_ufunc(
      lambda x: periodic_correlate(x, x) / spatial_axis_size, array,
      input_core_dims=[[spatial_axis]],
      output_core_dims=[[out_axis_name]])
  # we only report the unique half of the autocorrelation.
  num_unique_displacements = spatial_axis_size // 2
  result = full_result.isel({out_axis_name: slice(0, num_unique_displacements)})
  displacement_coords = array.coords[spatial_axis][:num_unique_displacements]
  result.coords[out_axis_name] = (out_axis_name, displacement_coords)
  return result


def energy_spectrum(ds: xarray.Dataset) -> xarray.DataArray:
  """Calculate the kinetic energy spectra at each wavenumber.

  Args:
    ds: dataset with `u`, `v` and/or `w` velocity components and corresponding
      spatial dimensions.

  Returns:
    Energy spectra as a function of wavenumber instead of space.
  """
  ndim = sum(dim in ds.dims for dim in 'xyz')
  velocity_components = list(XR_VELOCITY_NAMES[:ndim])
  fourier_ds = ds[velocity_components].map(fourier_transform)
  return kinetic_energy(fourier_ds)


def velocity_spatial_correlation(
    ds: xarray.Dataset,
    axis: str
) ->xarray.Dataset:
  """Computes velocity correlation along `axis` for all velocity components."""
  ndim = sum(dim in ds.dims for dim in 'xyz')
  velocity_components = list(XR_VELOCITY_NAMES[:ndim])
  correlation_fn = lambda x: spatial_autocorrelation(x, axis)
  correlations = ds[velocity_components].map(correlation_fn)
  name_mapping = {u: '_'.join([u, axis, 'correlation'])
                  for u in velocity_components}
  return correlations.rename(name_mapping)


def normalize(array: xarray.DataArray, state_dims: Tuple[str, ...]):
  """Returns `array` with slices along `state_dims` normalized to unity."""
  norm = np.sqrt((array ** 2).sum(state_dims))
  return array / norm
