use nalgebra::{Vector3, vector};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use rand::prelude::*;
use std::f32::consts::PI;
use rayon::prelude::*;

fn main() {
    

    //World
    let mut world = HittableCollection::new();

    let material_ground: Arc::<dyn Material> = Arc::new(Lambertian::new(&vector![0.5,0.5,0.5]));
    world.push(Arc::new(Sphere::new(vector![0.0,-1000.0,-1.0], 1000.0,  &(material_ground))));

    let mut rng = rand::thread_rng();

    for a in -11..11 {
        for b in -11..11 {
            let center = vector![(a as f32) + 0.9 * rng.gen::<f32>(), 0.2, (b as f32) + 0.9 * rng.gen::<f32>()];

            if (center - &vector![4.0,0.2,0.0]).norm_squared() > 0.9 {
                let mat_choose: f32 = rng.gen();
                if mat_choose < 0.8 {
                    //Diffuse
                    let albedo = get_random_vec(&mut rng).component_mul(&get_random_vec(&mut rng));
                    let material: Arc::<dyn Material> = Arc::new(Lambertian::new(&albedo));
                    world.push(Arc::new(Sphere::new(center, 0.2, &material)));
                } else if mat_choose < 0.95 {
                    //Metal
                    let albedo = get_random_vec_range(&mut rng, 1.0, 0.5);
                    let fuzz = rng.gen_range(0.0..0.5);
                    let material: Arc::<dyn Material> = Arc::new(Metal::new(&albedo,fuzz));
                    world.push(Arc::new(Sphere::new(center, 0.2, &material)));
                } else {
                    //Glass
                    let material: Arc::<dyn Material> = Arc::new(Dielectric::new(1.5));
                    world.push(Arc::new(Sphere::new(center, 0.2, &material)));
                }
            }
        }
    }

    let material1: Arc::<dyn Material> = Arc::new(Dielectric::new(1.5));
    let material2: Arc::<dyn Material> = Arc::new(Lambertian::new(&vector![0.4,0.2,0.1]));
    let material3: Arc::<dyn Material> = Arc::new(Metal::new(&vector![0.7,0.6,0.5], 0.0));

    world.push(Arc::new(Sphere::new(vector![0.0,1.0,0.0], 1.0, &material1)));
    world.push(Arc::new(Sphere::new(vector![-4.0,1.0,0.0], -1.0, &material2)));
    world.push(Arc::new(Sphere::new(vector![4.0, 1.0,0.0], 0.5, &material3)));
    
    
    let camera = Camera::new(
        16.0 / 9.0, 
        1200,
        500,
        50,
        20.0,
        vector![13.0,2.0,3.0], 
        vector![0.0,0.0,0.0], 
        vector![0.0,1.0,0.0],
        0.6,
        10.0
    );
    
    camera.render(&world);
}

type Point = Vector3<f32>;
type Direction = Vector3<f32>;
type Color = Vector3<f32>;

/// Simple line in 3D space
#[derive(Default)]
struct Ray {
    pub origin: Point,
    pub direction: Direction,
}

impl Ray {
    fn new(origin: Point, direction: Direction) -> Self {
        Self {origin, direction}
    }

    fn at(&self, t: f32) -> Point {
        self.origin + (t * self.direction)
    }
}

trait Hittable: Sync + Send {
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord, material: &mut Option<Arc<dyn Material>>) -> bool;
}

struct HittableCollection {
    objects: Vec<Arc<dyn Hittable>>,
}

impl HittableCollection {
    fn new() -> Self {
        let objects = Vec::new();
        Self{objects}
    }

    fn push(&mut self, object:  Arc<dyn Hittable>) {
        self.objects.push(object);
    }

    fn _clear(&mut self) {
        self.objects.clear();
    }
}

impl Hittable for HittableCollection {
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord, material: &mut Option<Arc<dyn Material>>) -> bool {
        let mut temp_rec = HitRecord::default();
        let mut hit_anything = false;
        let mut closest_current = t_interval.max;

        *material = None;

        self.objects.iter().for_each(|x| {
            if x.hit(ray,Interval::new(t_interval.min,closest_current), &mut temp_rec, material) {
                hit_anything = true;
                closest_current = temp_rec.t;
                *rec = temp_rec.clone();
            }
        });

        hit_anything
    }
}

struct Sphere {
    pub center: Point,
    pub radius: f32,
    pub material: Arc<dyn Material>,
}

impl Sphere {
    fn new(center: Point, radius: f32, material: &Arc<dyn Material>) -> Self {
        let material = Arc::clone(material);
        Self {center, radius, material}
    }
}

impl Hittable for Sphere
{
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord, material: &mut Option<Arc<dyn Material>>) -> bool {
        let dif = ray.origin.clone() - self.center;

        let a = ray.direction.norm_squared();
        let half_b = ray.direction.dot(&dif);
        let c = dif.norm_squared() - (self.radius * self.radius);

        let disc = half_b * half_b - a * c;
        if disc < 0.0 {return false;}

        let mut root = (-half_b - disc.sqrt()) / a;
        if !(t_interval.surrounds(root)){
            root = (-half_b + disc.sqrt()) / a;
            if !(t_interval.surrounds(root)){
                return false;
            }
        }

        rec.t = root;
        rec.point = ray.at(rec.t);
        let outward_normal = (rec.point - self.center) / self.radius;
        rec.set_face_normal(ray, &outward_normal);

        *material = Some(Arc::clone(&self.material));

        return true;
    }
}
#[derive(Clone,Default)]
struct HitRecord {
    pub point: Point,
    pub normal: Direction,
    //pub material: Option<Weak<dyn Material>>,
    pub t: f32,
    pub front_face: bool,
}

//struct  HitObserver(HitRecord,Option<Weak<dyn Material>>);

impl HitRecord {

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Direction) {
        //Sets hit record for normal vector

        self.front_face = ray.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {outward_normal.clone()} else {-outward_normal.clone()};
    }
}

fn color_to_rgb(color: Color) -> image::Rgb<u8> {

    //let result: Vector3<f32> = colors.iter().fold(Vector3::<f32>::default(), |acc, x| acc + x) / (colors.len() as f32);

    let mut result = color;
    result.iter_mut().for_each(|x: &mut f32| linear_to_gamma(x));

    static intensity: Interval = Interval{min:0.0, max:0.999};

    image::Rgb([
        (256.0 * intensity.clamp(result[0])) as u8,
        (256.0 * intensity.clamp(result[1])) as u8,
        (256.0 * intensity.clamp(result[2])) as u8,
    ])
}



struct Interval {
    pub min: f32,
    pub max: f32,
}

impl Interval {
    fn new(min: f32, max: f32) -> Self {
        Self{min, max}
    }

    fn default() -> Self {
        let min = f32::INFINITY;
        let max = -f32::INFINITY;
        Self{min,max}
    }

    fn contains(&self, x: f32) -> bool {
        (self.min <= x) && (x <= self.max)
    }

    fn surrounds(&self, x: f32) -> bool {
        (self.min < x) && (x < self.max)
    }

    fn clamp(&self, x: f32) -> f32 {
        if x < self.min {self.min}
        else if x > self.max {self.max}
        else {x}
    }
}

const empty: Interval =  Interval{min: f32::INFINITY, max:-f32::INFINITY};
const universal: Interval =  Interval{min:-f32::INFINITY, max:f32::INFINITY};

struct Basis(Direction,Direction,Direction);

struct Camera {
    //Image 
    aspect_ratio: f32,
    image_width: u32,
    image_height: u32,
    center: Vector3<f32>,
    pixel_start: Vector3<f32>,
    pixel_delta_u: Vector3<f32>,
    pixel_delta_v: Vector3<f32>,
    samples_per_pixel: usize,
    //rng: ThreadRng,
    max_depth: usize,

    v_fov: f32, //Vertical field of view in degrees
    look_from: Direction,
    look_at: Direction,
    v_up: Direction,
    basis: Basis,

    defocus_angle: f32,
    focus_dist: f32,
    defocus_disk_u: Direction,
    defocus_disk_v: Direction,
}

impl Camera {
    fn new(aspect_ratio: f32, image_width: u32, samples_per_pixel: usize, max_depth: usize, v_fov: f32, look_from: Direction, look_at: Direction, v_up: Direction, defocus_angle: f32, focus_dist:f32) -> Self {
        //Calculate image height: 1) round down 2) ensure at least 1
        let image_height = (image_width as f32/ aspect_ratio) as i32;
        let image_height: u32 = if image_height < 1 {1} else {image_height as u32};

        let center = look_from.clone();

        let dif = look_from.clone() - &look_at;

        //Determine viewport dimensions
        //let focal_length = dif.norm();
        let theta = v_fov * PI / 180.0;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * focus_dist;
        let viewport_width = viewport_height * ((image_width as f32) / (image_height as f32));

        let w = vec_unit(&dif);
        let u = vec_unit(&v_up.clone().cross(&w));
        let v = w.clone().cross(&u);

        let basis = Basis(w,u,v);


        //Calculate vectors across the horizontal and down vertical viewport edges
        let viewport_u: Point = viewport_width * basis.1;
        let viewport_v: Point = viewport_height * -basis.2;

        //Calculate horizontal and vertical delta (pixel size)
        let pixel_delta_u: Direction = viewport_u / (image_width as f32);
        let pixel_delta_v: Direction = viewport_v / (image_height as f32);

        //Calculate location of upper left pixel
        let viewport_start = 
            center.clone()
            - (focus_dist * basis.0)
            - (viewport_u / 2.0)
            - (viewport_v / 2.0);

        //Calculate camera defoucs disk basis
        let defocus_radius = focus_dist * (defocus_angle / 2.0 * PI / 180.0).tan();
        let defocus_disk_u = basis.1 * defocus_radius;
        let defocus_disk_v = basis.2 * defocus_radius;
    
        let pixel_start =
            viewport_start + 0.5 * (pixel_delta_u + pixel_delta_v);


        Self {aspect_ratio, image_width, image_height, center, pixel_start, pixel_delta_u, pixel_delta_v, samples_per_pixel, max_depth, v_fov, look_from,look_at,v_up, basis, defocus_angle, focus_dist, defocus_disk_u,defocus_disk_v}
    }

    fn render(&self, world: &dyn Hittable) {

        let mut imgbuf = image::ImageBuffer::new(self.image_width, self.image_height);
        
        println!("Started generating image of size {} * {}",self.image_width, self.image_height);

        let camera = Arc::new(self);

        //for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        imgbuf.par_enumerate_pixels_mut().for_each( |(x, y, pixel)| {
            //if x == 0 //&& y % 10 == 0 {
            {
                //println!("({} {})",x,y);
            }

            let mut rng = rand::thread_rng();

            let result = (0..(camera.samples_per_pixel)).map(|_| {
                let ray = self.get_ray(x, y, &mut rng);
                self.ray_color(self.max_depth, &ray, world, &mut rng)
            }).fold(Vector3::<f32>::default(), |acc, x| acc + x) 
            / (self.samples_per_pixel as f32);

            *pixel = color_to_rgb(result);
        });

        imgbuf.save("image.png").unwrap();

        println!("Finished generating");
}

    
    fn ray_color(&self, depth: usize, ray: &Ray, hittable: &dyn Hittable, rng: &mut ThreadRng) -> Color {
        let mut hit_record = HitRecord::default();

        if depth <= 0 {
            return vector![0.0, 0.0, 0.0];
        }

        let mut material: Option<Arc<dyn Material>> = None;

        if hittable.hit(ray, Interval::new(0.001, f32::INFINITY), &mut hit_record, &mut material) {
            let mut scattered = Ray::default();
            let mut attenuation = Color::default();

            let hit_record = hit_record;

            if material.unwrap().scatter(ray, &hit_record, &mut attenuation, &mut scattered, rng) {
                let temp = &self.ray_color(depth - 1, &scattered, hittable, rng);
                return attenuation.component_mul(temp);
                
            } else {
                return vector![0.0,0.0,0.0];
            };
        };

        let unit_direction = ray.direction / ray.direction.norm();
        let a = 0.5 * (unit_direction.y + 1.0);
        vector![1.0,1.0,1.0].lerp(&vector![0.5,0.7,1.0], a)
    } 

    ///Get randomly sample ray from camera at pixel location, originating from camera defocus disk
    fn get_ray(&self, i: u32, j: u32, rng: &mut ThreadRng) -> Ray {
        let pixel_center = self.pixel_start + ((i as f32) * self.pixel_delta_u + (j as f32) * self.pixel_delta_v);
        let pixel_sample = pixel_center + self.pixel_sample_square(rng);

        let ray_origin = if self.defocus_angle <= 0.0 {self.center} else {self.defocus_disk_sample(rng)};
        let ray_direction = pixel_sample - ray_origin;

        Ray::new(ray_origin, ray_direction)
    }

    ///Returns a random point in square surrounding origin of pixel
    fn pixel_sample_square(&self, rng: &mut ThreadRng) -> Vector3<f32> {
        let px = -0.5 + rng.gen::<f32>();
        let py = -0.5 + rng.gen::<f32>();
        px * self.pixel_delta_u + py * self.pixel_delta_v
    }

    fn defocus_disk_sample(&self, rng: &mut ThreadRng) -> Direction {
        let p = get_random_in_unit_disk(rng);
        self.center + p[0] * &self.defocus_disk_u + p[1] * &self.defocus_disk_v
    }
}


fn get_random_vec(rng: &mut ThreadRng) -> Vector3<f32> {
    vector![rng.gen(), rng.gen(), rng.gen()]
}

fn get_random_vec_range(rng: &mut ThreadRng, max: f32, min:f32) -> Direction {
    vector![rng.gen_range(min..max), rng.gen_range(min..max), rng.gen_range(min..max)]
}

fn get_random_unit(rng: &mut ThreadRng) -> Vector3<f32> {
    loop {
        let p = get_random_vec(rng);
        if p.norm() < 1.0 {
            return p
        }
    }
}

fn get_random_vec_hemi(rng: &mut ThreadRng, normal: &Vector3<f32>) -> Vector3<f32> {
    let on_unit_sphere = get_random_unit(rng);
    if on_unit_sphere.dot(normal) > 0.0 {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
}

fn get_random_in_unit_disk(rng: &mut ThreadRng) -> Direction {
    loop {
        let p = vector![rng.gen_range(-1.0..1.0),rng.gen_range(-1.0..1.0),0.0];
        if p.norm_squared() < 1.0 {return p;};
    }
}

fn linear_to_gamma(component: &mut f32) {
    *component = component.sqrt();
}

trait Material: Sync + Send {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool;
}

struct Lambertian {
    albedo: Color,
    //rng: ThreadRng,
}

impl Lambertian {
    fn new(albedo: &Color) -> Self {
        let albedo = albedo.clone();
        Self {albedo}
    }
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit_record: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool {
        let mut scatter_direction = hit_record.normal + get_random_unit(rng);

        //Catch degenerate scatter direction
        let s = 1e-8;
        if scatter_direction.norm() < s {
            scatter_direction = hit_record.normal;
        }

        *scattered = Ray::new(hit_record.point, scatter_direction);
        *attenuation = self.albedo;
        true
    }
}

fn reflect(v: &Direction, n: &Direction) -> Direction {
    v.clone() - 2.0 * v.dot(n) * n
}

fn refract(vec_in: &Direction, normal: &Direction, coeff_in: f32, coeff_out: f32) -> Direction{
    let ratio = coeff_in / coeff_out;
    let cos_theta = (vec_in.dot(normal)).min(1.0);
    let vec_out_perp = ratio * (vec_in - cos_theta * normal);
    let vec_out_para = -(1.0 - vec_out_perp.norm_squared()).sqrt() * normal;
    vec_out_perp + vec_out_para
}

fn vec_unit(v: &Direction) -> Direction {
    let v = v / v.norm();
    v
}

struct Metal {
    albedo: Color,
    fuzz: f32,
}

impl Metal {
    fn new(albedo: &Color, fuzz: f32) -> Self {
        let albedo = albedo.clone();
        Self{albedo, fuzz}
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool {
        let reflected = reflect(&vec_unit(&ray.direction), &hit_record.normal);
        *scattered = Ray::new(hit_record.point, reflected + self.fuzz * get_random_unit(rng));
        *attenuation = self.albedo;
        scattered.direction.dot(&hit_record.normal) > 0.0
    }
}

struct Dielectric {
    index_of_refraction: f32,
}

impl Dielectric {
    fn new(index_of_refraction: f32) -> Self {
        Self {index_of_refraction}
    }

    fn reflectance(cosine: f32, ref_index: f32) -> f32 {
        let mut r0 = (1.0 - ref_index) / (1.0 + ref_index);
        r0 = r0 * r0;
        r0 * (1.0 - r0) * (1.0 - cosine).powf(5.0)
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord, attenuation: &mut Color, scattered: &mut Ray, rng: &mut ThreadRng) -> bool {
        *attenuation = vector![1.0,1.0,1.0];
        let refraction_ratio = if hit_record.front_face {1.0 / self.index_of_refraction} else {self.index_of_refraction};

        let unit_direction = vec_unit(&ray.direction);
        
        let cos_theta = (-unit_direction.dot(&hit_record.normal)).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        
        let direction = 
            if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > rng.gen(){
                reflect(&unit_direction, &hit_record.normal)
            } else {
                refract(&unit_direction, &hit_record.normal, refraction_ratio, 1.0)
            };
        
        *scattered = Ray::new(hit_record.point, direction);

        true
    }
}

