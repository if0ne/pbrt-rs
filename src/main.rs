use std::primitive;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = false)]
    use_gpu: bool,

    #[arg(short, long, default_value_t = false)]
    wave_front: bool,
}

#[derive(Debug)]
struct Options {
    use_gpu: bool,
    wave_front: bool,
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

    fn check(&self) -> bool {
        self.bits & Self::PTR_MASK != 0
    }
}

#[derive(Debug)]
struct Primitive {
    ptr: TaggedPointer,
}

impl Primitive {
    fn bounds(&self) -> Bounds3 {
        Bounds3 {}
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
    fn preprocess(&mut self, bound: &Bounds3) {}
}

#[derive(Debug)]
struct Bounds3 {}

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
            Bounds3 {}
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

fn main() {
    let args = Args::parse();

    let options = Options {
        use_gpu: args.use_gpu,
        wave_front: args.wave_front,
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
