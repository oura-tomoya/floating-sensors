#!/usr/bin/env python

"""
Layers for the velocity field estimation using floating sensors.

T.Oura, 2025
"""

import tensorflow as tf
import numpy as np

class ParticleTrackingLayer(tf.keras.layers.Layer):
    """
    The layer to track sensors in the ML model.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):    
        xpyp_n = tf.cast(inputs[0][:, :, :2], tf.float64) # Particle locations at t = n * dt
        ufvf_n_estimated = tf.cast(inputs[1], tf.float64) # Estimated velocity fields at t = n * dt

        nx = tf.shape(inputs[1])[1]
        ny = tf.shape(inputs[1])[2]
        x_grid = tf.linspace(0.0, 2.0 * np.pi, nx + 1) # [0, 2pi]
        y_grid = tf.linspace(0.0, 2.0 * np.pi, ny + 1) # [0, 2pi]
        x_grid = tf.cast(x_grid[0:nx], tf.float64)     # [0, 2pi)
        y_grid = tf.cast(x_grid[0:ny], tf.float64)     # [0, 2pi)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        Lx = x_grid[-1] + dx
        Ly = y_grid[-1] + dy
        dt = tf.cast(1e-2, tf.float64)
        
        batch_size = tf.cast(tf.shape(xpyp_n)[0], tf.int32) # The number of temporal snapshots
        num_sen = tf.cast(tf.shape(xpyp_n)[1], tf.int32)    # The number of sensors
        
        # --------------------------------------------------------------------- #
        
        @tf.function
        def bilinear_interpolation(xpyp, ufvf):
            # Cell indices (ix1, iy1) where the sensor is located.
            ix1 = tf.cast(xpyp[:, :, 0] / dx, dtype=tf.int32)
            iy1 = tf.cast(xpyp[:, :, 1] / dy, dtype=tf.int32)

            # Calculate the cell indices (ix2, iy2) adjacent to (ix1, iy1).
            # The priodic boundary condition is applied.
            # NOTE: int(ix1 / (nx-1)) is 1 if ix1 > nx, else 0.
            ix2 = ix1 + 1 - tf.cast(ix1 / (nx - 1), dtype=tf.int32) * nx
            iy2 = iy1 + 1 - tf.cast(iy1 / (ny - 1), dtype=tf.int32) * ny

            # `it` is the matrix which contains time indices as follows.
            #   [[  0,   0,   0, ...,   0,   0,   0],
            #    [  1,   1,   1, ...,   1,   1,   1],
            #    [  2,   2,   2, ...,   2,   2,   2],
            #          ...
            #    ]
            # The size of `it` is equal to that of `ix1` and `iy1`.
            it = tf.tile(
                tf.expand_dims(tf.range(batch_size), axis=-1), 
                [1, num_sen])

            # Tensors of stencil indices (it, ix, iy) around the sensor locations.
            # NOTE: The 3rd element is the vector of (it, ix, iy).
            itxy11 = tf.stack([it, ix1, iy1], axis=-1)
            itxy12 = tf.stack([it, ix1, iy2], axis=-1)
            itxy21 = tf.stack([it, ix2, iy1], axis=-1)
            itxy22 = tf.stack([it, ix2, iy2], axis=-1)

            # Fluid velocities in the 2x2 stencils.
            uf11 = tf.gather_nd(ufvf[:, :, :, 0], itxy11)
            uf12 = tf.gather_nd(ufvf[:, :, :, 0], itxy12)
            uf21 = tf.gather_nd(ufvf[:, :, :, 0], itxy21)
            uf22 = tf.gather_nd(ufvf[:, :, :, 0], itxy22)

            vf11 = tf.gather_nd(ufvf[:, :, :, 1], itxy11)
            vf12 = tf.gather_nd(ufvf[:, :, :, 1], itxy12)
            vf21 = tf.gather_nd(ufvf[:, :, :, 1], itxy21)
            vf22 = tf.gather_nd(ufvf[:, :, :, 1], itxy22)

            # Coefficient (alpha and beta) of the bilinear interpolation.
            alpha = (xpyp[:, :, 0] - tf.gather(x_grid, ix1)) / dx
            beta  = (xpyp[:, :, 1] - tf.gather(y_grid, iy1)) / dy
            
            # Bilinear interpolation at the sensor locations.
            uf_interp = (
                tf.math.multiply(tf.math.multiply(1.0 - alpha, 1.0 - beta), uf11) + 
                tf.math.multiply(tf.math.multiply(alpha, 1.0 - beta), uf21) +
                tf.math.multiply(tf.math.multiply(alpha, beta), uf22) +
                tf.math.multiply(tf.math.multiply(1.0 - alpha, beta), uf12)
                )

            vf_interp = (
                tf.math.multiply(tf.math.multiply(1.0 - alpha, 1.0 - beta), vf11) + 
                tf.math.multiply(tf.math.multiply(alpha, 1.0 - beta), vf21) +
                tf.math.multiply(tf.math.multiply(alpha, beta), vf22) +
                tf.math.multiply(tf.math.multiply(1.0 - alpha, beta), vf12)
                )

            return uf_interp, vf_interp
        
        # --------------------------------------------------------------------- #
        
         # Interpolated fluid veloicties at the sensor locations at n.
        up_n, vp_n = \
            bilinear_interpolation(xpyp_n, ufvf_n_estimated)
        upvp_n = tf.stack([up_n, vp_n], axis=-1)
        
        # Time marching w/ the Euler explicit method.
        xpyp_n_1 = xpyp_n + upvp_n * dt
        
        return xpyp_n_1[:, :, 0], xpyp_n_1[:, :, 1]

class PeriodicPaddingLayer(tf.keras.layers.Layer):
    """
    Two dimensional periodic padding.
    """
    def __init__(self, padding, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding
    
    def call(self, inputs):
        input_shape = tf.shape(inputs)

        padded = tf.concat(
            [inputs[:, -self.padding:, :, :],
             inputs, 
             inputs[:, :self.padding, :, :]], 
            axis=1)

        padded = tf.concat(
            [padded[:, :, -self.padding:, :], 
             padded, 
             padded[:, :, :self.padding, :]],
            axis=2)

        return padded
