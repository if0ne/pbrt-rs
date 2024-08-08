use clap::Parser;
use std::cell::RefCell;

trait Number:
    std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Mul<Output = Self>
    + Copy
    + Default
    + PartialEq
    + Sized
{
    fn zero() -> Self;
    fn neutral() -> Self;
}

impl Number for f32 {
    fn zero() -> Self {
        0.0
    }

    fn neutral() -> Self {
        1.0
    }
}

impl Number for u32 {
    fn zero() -> Self {
        0
    }

    fn neutral() -> Self {
        1
    }
}

impl Number for i32 {
    fn zero() -> Self {
        0
    }

    fn neutral() -> Self {
        1
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    use_gpu: bool,

    #[arg(short, long, default_value_t = false)]
    wave_front: bool,

    #[arg(short, long, default_value_t = false)]
    quiet: bool,
}

#[derive(Clone, Debug, Default)]
struct Options {
    use_gpu: bool,
    wave_front: bool,
    quiet: bool,
}

#[derive(Debug)]
struct BasicScene {}

#[derive(Debug)]
struct BasicSceneBuilder<'a> {
    scene: &'a mut BasicScene,
}

#[derive(Debug)]
struct TaggedPointer {
    bits: u64,
}

impl TaggedPointer {
    const TAG_SHIFT: u64 = 57;
    const TAG_BITS: u64 = 64 - Self::TAG_SHIFT;
    const TAG_MASK: u64 = ((1 << Self::TAG_BITS) - 1) << Self::TAG_SHIFT;
    const PTR_MASK: u64 = !Self::TAG_MASK;

    const fn check(&self) -> bool {
        self.bits & Self::PTR_MASK != 0
    }
}

#[derive(Debug)]
struct Primitive {
    ptr: TaggedPointer,
}

impl Primitive {
    fn bounds(&self) -> Bounds<3, f32> {
        todo!()
    }

    fn check(&self) -> bool {
        self.ptr.check()
    }

    fn intersect(&self, ray: &Ray, tmax: f32) -> Option<ShapeIntersection> {
        None
    }

    fn intersect_p(&self, ray: &Ray, tmax: f32) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LightType {
    DeltaDirection,
    Infinite,
}

#[derive(Clone, Debug)]
struct Light {
    r#type: LightType,
}

impl Light {
    fn preprocess(&mut self, bound: &Bounds<3, f32>) {}
}

#[derive(Clone, Copy, Debug)]
struct Bounds<const N: usize, T: Number> {
    bounds: [T; N],
}

#[derive(Clone, Copy, Debug)]
struct BoundsIterator<const N: usize, T: Number> {
    bounds: [T; N],
    point: [T; N],
}

#[derive(Clone, Copy, Debug)]
struct Point<const N: usize, T: Number> {
    point: [T; N],
}

impl<const N: usize, T: Number> Bounds<N, T> {
    fn area(&self) -> T {
        let mut prod = T::neutral();

        for i in self.bounds {
            prod = prod * i;
        }

        prod
    }
}

impl<T: Number> IntoIterator for Bounds<2, T> {
    type Item = Point<2, T>;
    type IntoIter = BoundsIterator<2, T>;

    fn into_iter(self) -> Self::IntoIter {
        BoundsIterator {
            bounds: self.bounds,
            point: Default::default(),
        }
    }
}

impl<T: Number> Iterator for BoundsIterator<2, T> {
    type Item = Point<2, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.point[0] == self.bounds[0] && self.point[1] == self.bounds[1] {
            return None;
        }

        if self.point[0] == self.bounds[0] {
            self.point[0] = T::zero();
            self.point[1] += T::neutral();

            return Some(Point { point: self.point });
        }

        self.point[0] += T::neutral();
        Some(Point { point: self.point })
    }
}

#[derive(Debug)]
struct ShapeIntersection {}

#[derive(Debug)]
struct Ray {}

#[derive(Debug)]
struct BaseIntegrator {
    aggregate: Primitive,
    lights: Vec<Light>,
    infinity_lights: Vec<Light>,
}

impl BaseIntegrator {
    fn new(aggregate: Primitive, mut lights: Vec<Light>) -> Self {
        let scene_bounds = if aggregate.check() {
            aggregate.bounds()
        } else {
            todo!()
        };

        let mut infinity_lights = vec![];

        for light in &mut lights {
            light.preprocess(&scene_bounds);

            if light.r#type == LightType::Infinite {
                infinity_lights.push(light.clone());
            }
        }

        Self {
            aggregate,
            lights,
            infinity_lights,
        }
    }

    fn intersect(&self, ray: &Ray, tmax: f32) -> Option<ShapeIntersection> {
        if self.aggregate.check() {
            self.aggregate.intersect(ray, tmax)
        } else {
            None
        }
    }

    fn intersect_p(&self, ray: &Ray, tmax: f32) -> bool {
        if self.aggregate.check() {
            self.aggregate.intersect_p(ray, tmax)
        } else {
            false
        }
    }
}

trait Integrator {
    fn render(&self);
}

#[derive(Debug)]
struct Film {}

impl Film {
    fn pixel_bounds(&self) -> Bounds<2, i32> {
        todo!()
    }
}

#[derive(Debug)]
struct Camera {
    film: Film,
}

impl Camera {
    fn get_film(&self) -> &Film {
        &self.film
    }
}

#[derive(Clone, Debug)]
struct Sampler {}

impl Sampler {
    fn samples_per_pixel(&self) -> u32 {
        0
    }

    fn start_pixel_sample(&self, pixel: Point<2, i32>, index: u32) {}
}

#[derive(Debug)]
struct ProgressReport {}

impl ProgressReport {
    fn new(max_bound: u64, title: &str, is_quiet: bool) -> Self {
        Self {}
    }

    fn update(&mut self, progress: u64) {
        todo!()
    }
}

#[derive(Debug)]
struct ScratchBuffer {}

impl ScratchBuffer {
    fn reset(&mut self) {}
}

#[derive(Debug)]
struct BaseImageTileIntegrator {
    base: BaseIntegrator,
    camera: Camera,
    sampler: Sampler,

    options: Options,
}

trait ImageTileIntegrator: Integrator {
    fn evaluate_pixel_sample(
        &self,
        pixel: Point<2, i32>,
        sample_index: u32,
        sampler: &mut Sampler,
        scratch_buffer: &mut ScratchBuffer,
    );
}

impl ImageTileIntegrator for BaseImageTileIntegrator {
    fn evaluate_pixel_sample(
        &self,
        pixel: Point<2, i32>,
        sample_index: u32,
        sampler: &mut Sampler,
        scratch_buffer: &mut ScratchBuffer,
    ) {
    }
}

impl BaseImageTileIntegrator {
    thread_local! {
        static scratch_buffer: RefCell<ScratchBuffer> = RefCell::new(ScratchBuffer {});
    }

    thread_local! {
        static sampler: RefCell<Sampler> = RefCell::new(Sampler {});
    }

    fn new(
        camera: Camera,
        sampler: Sampler,
        aggregate: Primitive,
        lights: Vec<Light>,
        options: Options,
    ) -> Self {
        let base = BaseIntegrator::new(aggregate, lights);

        Self {
            base,
            camera,
            sampler,
            options,
        }
    }

    fn intersect(&self, ray: &Ray, tmax: f32) -> Option<ShapeIntersection> {
        self.base.intersect(ray, tmax)
    }

    fn intersect_p(&self, ray: &Ray, tmax: f32) -> bool {
        self.base.intersect_p(ray, tmax)
    }
}

impl Integrator for BaseImageTileIntegrator {
    fn render(&self) {
        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler.samples_per_pixel();
        let mut progress = ProgressReport::new(
            (spp as i32 * pixel_bounds.area()) as _,
            "Rendering",
            self.options.quiet,
        );

        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        while wave_start < spp {
            // Parallel render
            parallel_for2d(pixel_bounds, |tile_bounds| {
                let sampler = Self::sampler.with_borrow_mut(|sampler| {
                    Self::scratch_buffer.with_borrow_mut(|scratch_buffer| {
                        for pixel in tile_bounds {
                            for sample_index in wave_start..wave_end {
                                sampler.start_pixel_sample(pixel, sample_index);
                                self.evaluate_pixel_sample(
                                    pixel,
                                    sample_index,
                                    sampler,
                                    scratch_buffer,
                                );
                                scratch_buffer.reset();
                            }
                        }

                        progress
                            .update(((wave_end - wave_start) * tile_bounds.area() as u32) as u64);
                    });
                });
            });

            wave_start = wave_end;
            wave_end = spp.min(wave_end + next_wave_size);
            next_wave_size = 64.min(2 * next_wave_size);
        }
    }
}

fn main() {
    let args = Args::parse();

    let options = Options {
        use_gpu: args.use_gpu,
        wave_front: args.wave_front,
        quiet: args.quiet,
    };
    let filenames = Vec::<String>::new();

    // Process command-line arguments

    init_pbrt(&options);

    let mut scene = BasicScene {};
    let mut builder = BasicSceneBuilder { scene: &mut scene };
    parse_files(&mut builder, filenames);

    if options.use_gpu || options.wave_front {
        render_wavefront(&scene);
    } else {
        render_cpu(&scene);
    }

    cleanup_pbrt();
}

fn init_pbrt(options: &Options) {}

fn parse_files(scene_builder: &mut BasicSceneBuilder<'_>, filenames: Vec<String>) {}

fn render_cpu(scene: &BasicScene) {}

fn render_wavefront(scene: &BasicScene) {}

fn cleanup_pbrt() {}

fn parallel_for2d(size: Bounds<2, i32>, func: impl FnMut(Bounds<2, i32>)) {}
