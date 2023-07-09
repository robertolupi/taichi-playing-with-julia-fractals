import click
from pathlib import Path
import tempfile 

import taichi as ti
import taichi.math as tm


ARCH = {
    "cpu": ti.cpu,
    "gpu": ti.gpu,
    "cuda": ti.cuda,
    "metal": ti.metal,
    "opengl": ti.opengl,
    "vulkan": ti.vulkan,
}


@ti.func
def julia(z, c, M):
    k = 0
    while k < M and z.norm() < 20:
        z = tm.cmul(z, z) + c
        k += 1
    return k / M


@ti.kernel
def render(pixels: ti.template(), i: ti.f32):
    s = tm.normalize(tm.vec2(pixels.shape))
    for I in ti.grouped(pixels):
        uv = I / pixels.shape - s
        z = uv
        k = 0
        l = tm.vec3(
            1 - julia(z, tm.vec2(-0.82, tm.cos(0.2 * i)), 99),
            julia(z, tm.vec2(-0.82, tm.cos(0.33 * i)), 53),
            1 - julia(z, tm.vec2(-0.82, tm.cos(0.19 * i)), 11),
        )

        pixels[I] = l


@click.command()
@click.option(
    "--arch", "-a", type=click.Choice(ARCH.keys()), help="Architecture to use"
)
@click.option(
    "--res",
    "-r",
    default=(1920, 1080),
    type=(int, int),
    help="Resolution of the window (e.g. -r 1920 1080)",
)
@click.option(
    "--gif/--no-gif", help="Save output as gif", type=click.BOOL, default=False,
)
@click.option(
    "--mp4/--no-mp4", help="Save output as mp4", type=click.BOOL, default=False,
)
def main(arch, res, gif, mp4):
    ti.init(ARCH.get(arch))

    
    video_manager = None
    if gif or mp4:
        temp_dir = Path(tempfile.mkdtemp())
        video_manager = ti.tools.VideoManager(output_dir=temp_dir, framerate=24, automatic_build=False)

    pixels = ti.Vector.field(3, dtype=ti.f32, shape=res)
    window = ti.ui.Window("Playground", res)
    canvas = window.get_canvas()
    i = 0.0
    k = 0
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
        render(pixels, i)
        canvas.set_image(pixels)

        if video_manager:
            pixels_img = pixels.to_numpy()
            video_manager.write_frame(pixels_img)

        window.show()
        i += tm.pi / 100.0
        k += 1

    if video_manager:
        video_manager.make_video(gif=gif, mp4=mp4)
        if gif:
            Path(video_manager.get_output_filename(".gif")).rename(Path.cwd() / "output.gif")
        if mp4:
            Path(video_manager.get_output_filename(".mp4")).rename(Path.cwd() / "output.mp4")

if __name__ == "__main__":
    main()
