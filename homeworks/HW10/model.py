from typing import Callable

import jax
import jax.numpy as jnp

import equinox as eqx

from jaxtyping import PyTree, Array, Float, PRNGKeyArray



class DoubleConvBlock(eqx.Module):
    conv_in: eqx.nn.Conv2d
    conv_out: eqx.nn.Conv2d
    activation: Callable
    bn1: eqx.nn.BatchNorm
    bn2: eqx.nn.BatchNorm

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels = None,
        activation = jax.nn.relu,
        *,
        key
    ):
        if not mid_channels:
            mid_channels = out_channels

        key1, key2 = jax.random.split(key, 2)

        self.conv_in = eqx.nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            key=key1
        )
        self.bn1 = eqx.nn.BatchNorm(mid_channels, axis_name="batch")
        self.conv_out = eqx.nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=key2
        )
        self.bn2 = eqx.nn.BatchNorm(out_channels, axis_name="batch")
        self.activation = activation

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State
    ):
        x = self.conv_in(x)
        x, state = self.bn1(x, state)
        x = self.activation(x)
        x = self.conv_out(x)
        x, state = self.bn2(x, state)
        x = self.activation(x)
        return x, state


class DownBlock(eqx.Module):
    pool: eqx.nn.MaxPool2d
    double_conv: DoubleConvBlock

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        key
    ):
        self.double_conv = DoubleConvBlock(
            in_channels,
            out_channels,
            key=key
        )
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State
    ):
        x = self.pool(x)
        x, state = self.double_conv(x, state)
        return x, state


class UpBlock(eqx.Module):
    up: eqx.nn.ConvTranspose2d
    conv: DoubleConvBlock

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        key
    ):
        up_key, conv_key = jax.random.split(key, 2)
        self.up = eqx.nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
            key=up_key
        )
        self.conv = DoubleConvBlock(
            in_channels, out_channels, key=conv_key
        )

    def __call__(
        self,
        x1: Float[Array, "c h w"],
        x2: Float[Array, "c h w"],
        state: eqx.nn.State
    ):
        x1 = self.up(x1)
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]

        pad_width = (
            (0, 0),
            (diffY // 2, diffY - diffY // 2),
            (diffX // 2, diffX - diffX // 2)
        )
        x1 = jnp.pad(
            x1,
            pad_width=pad_width,
            mode='constant'
        )
        x = jnp.concatenate([x2, x1], axis=0)
        return self.conv(x, state)


class BilinearUpBlock(eqx.Module):
    conv: DoubleConvBlock

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        key
    ):
        self.conv = DoubleConvBlock(
            in_channels, out_channels, key=key
        )

    def __call__(
        self,
        x1: Float[Array, "c h w"],
        x2: Float[Array, "c h w"],
        state: eqx.nn.State
    ):
        # upsampling
        x1 = jnp.permute_dims(x1, axes=(1, 2, 0))
        h, w, c = x1.shape
        x1 = jax.image.resize(
            x1,
            shape=(2 * h, 2 * w, c),
            method='bilinear'
        )
        x1 = jnp.permute_dims(x1, axes=(2, 0, 1))
        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]

        pad_width = (
            (0, 0),
            (diffY // 2, diffY - diffY // 2),
            (diffX // 2, diffX - diffX // 2)
        )
        x1 = jnp.pad(
            x1,
            pad_width=pad_width,
            mode='constant'
        )
        x = jnp.concatenate([x2, x1], axis=0)
        return self.conv(x, state)

class OutConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        key
    ):
        self.conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, key=key)

    def __call__(
        self,
        x: Float[Array, "c h w"],
    ):
        return self.conv(x)

class UNet(eqx.Module):
    inc: DoubleConvBlock
    down1: DownBlock
    down2: DownBlock
    down3: DownBlock
    down4: DownBlock
    up1: UpBlock | BilinearUpBlock
    up2: UpBlock | BilinearUpBlock
    up3: UpBlock | BilinearUpBlock
    up4: UpBlock | BilinearUpBlock
    outc: OutConvBlock

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        *,
        key
    ):
        key_inc, key_up, key_down, key_outc = jax.random.split(key, 4)
        key_ups = jax.random.split(key_up, 4)
        key_downs = jax.random.split(key_down, 4)

        self.inc = DoubleConvBlock(in_channels, 32, key=key_inc)
        self.down1 = DownBlock(32, 64, key=key_downs[0])
        self.down2 = DownBlock(64, 128, key=key_downs[1])
        self.down3 = DownBlock(128, 256, key=key_downs[2])
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(256, 512 // factor, key=key_downs[3])

        if bilinear:
            self.up1 = BilinearUpBlock(512, 128, key=key_ups[0])
            self.up2 = BilinearUpBlock(256, 64, key=key_ups[1])
            self.up3 = BilinearUpBlock(128, 32, key=key_ups[2])
            self.up4 = BilinearUpBlock(64, 32, key=key_ups[3])
        else:
            self.up1 = UpBlock(512, 256, key=key_ups[0])
            self.up2 = UpBlock(256, 128, key=key_ups[1])
            self.up3 = UpBlock(128, 64, key=key_ups[2])
            self.up4 = UpBlock(64, 32, key=key_ups[3])

        self.outc = OutConvBlock(32, out_channels, key=key_outc)

    def __call__(
        self,
        x: Float[Array, "c h w"],
        state: eqx.nn.State
    ):
        x1, state = self.inc(x, state)
        x2, state = self.down1(x1, state)
        x3, state = self.down2(x2, state)
        x4, state = self.down3(x3, state)
        x, state = self.down4(x4, state)
        x, state = self.up1(x, x4, state)
        x, state = self.up2(x, x3, state)
        x, state = self.up3(x, x2, state)
        x, state = self.up4(x, x1, state)
        out = self.outc(x)
        return out, state