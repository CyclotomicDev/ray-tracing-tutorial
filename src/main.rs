use nalgebra::{Vector3, vector};
use std::rc::Rc;
use rand::prelude::*;


fn main() {
    

    //World
    let mut world = HittableCollection::new();
    world.push(Rc::new(Sphere::new(vector![0.0,0.0,-1.0], 0.5)));
    world.push(Rc::new(Sphere::new(vector![0.0,-100.5,-1.0], 100.0)));
    
    let mut camera = Camera::new(16.0 / 9.0, 400);
    
    camera.render(&world);
}

type Point = Vector3<f32>;
type Direction = Vector3<f32>;
type Color = Vector3<f32>;

/// Simple line in 3D space
/// 
/// #Examples
/// 
/// ```
/// let line = Ray::new(vector![1.0,2.0,3.0], vector![1.0,0.0,1.0]);
/// 
/// assert_eq!(vector![3.0, 2.0, 5.0], line.at(2.0));
/// ```
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

trait Hittable {
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord) -> bool;
}

struct HittableCollection {
    objects: Vec<Rc<dyn Hittable>>,
}

impl HittableCollection {
    fn new() -> Self {
        let objects = Vec::new();
        Self{objects}
    }

    fn push(&mut self, object:  Rc<dyn Hittable>) {
        self.objects.push(object);
    }

    fn clear(&mut self) {
        self.objects.clear();
    }
}

impl Hittable for HittableCollection {
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord) -> bool {
        let mut temp_rec = HitRecord::default();
        let mut hit_anything = false;
        let mut closest_current = t_interval.max;

        self.objects.iter().for_each(|x| {
            if x.hit(ray,Interval::new(t_interval.min,closest_current), &mut temp_rec) {
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
}

impl Sphere {
    fn new(center: Point, radius: f32) -> Self {
        Self {center, radius}
    }

    /*
    ///Returns value of closest intersection, if exists
    fn intersection(&self, ray: &Ray) -> Option<f32> {
        let dif = ray.origin.clone() - self.center;
        let a = ray.direction.norm_squared();
        let half_b = ray.direction.dot(&dif);
        let c = dif.norm_squared() - self.radius * self.radius;

        let disc = half_b * half_b - a * c;

        if disc >= 0.0 {
            Some((-half_b - disc.sqrt()) / a)
        } else {
            None
        }
    }

    //Calculates unit vector (assuming given point on surface)
    fn unit_vector<'a>(&'a self, point: &'a mut Point) -> &mut Point {
        *point = (*point - self.center) / self.radius;
        point
    }*/
}

impl Hittable for Sphere
{
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord) -> bool {
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

        return true;
    }
}
#[derive(Clone,Default)]
struct HitRecord {
    pub point: Point,
    pub normal: Direction,
    pub t: f32,
    pub front_face: bool,
}

impl HitRecord {

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Direction) {
        //Sets hit record for normal vector

        self.front_face = ray.direction.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {outward_normal.clone()} else {-outward_normal.clone()};
    }
}

fn color_to_rgb(color: Color) -> image::Rgb<u8> {

    //let result: Vector3<f32> = colors.iter().fold(Vector3::<f32>::default(), |acc, x| acc + x) / (colors.len() as f32);

    let result = color;

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
    rng: ThreadRng,
}

impl Camera {
    fn new(aspect_ratio: f32, image_width: u32) -> Self {
        //Calculate image height: 1) round down 2) ensure at least 1
        let image_height = (image_width as f32/ aspect_ratio) as i32;
        let image_height: u32 = if image_height < 1 {1} else {image_height as u32};

        let center = vector![0.0,0.0,0.0];

        //Determine viewport dimensions
        let focal_length = 1.0;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * ((image_width as f32) / (image_height as f32));

        //Calculate vectors across the horizontal and down vertical viewport edges
        let viewport_u: Point = vector![viewport_width, 0.0, 0.0];
        let viewport_v: Point = vector![0.0, -viewport_height, 0.0];

        //Calculate horizontal and vertical delta (pixel size)
        let pixel_delta_u: Direction = viewport_u / (image_width as f32);
        let pixel_delta_v: Direction = viewport_v / (image_height as f32);

        //Calculate location of upper left pixel
        let viewport_start = 
            center.clone()
            - vector![0.0, 0.0, focal_length]
            - (viewport_u / 2.0)
            - (viewport_v / 2.0);
    
        let pixel_start =
            viewport_start + 0.5 * (pixel_delta_u + pixel_delta_v);

        let samples_per_pixel = 100;

        let rng = rand::thread_rng();

        Self {aspect_ratio, image_width, image_height, center, pixel_start, pixel_delta_u, pixel_delta_v, samples_per_pixel, rng}
    }

    fn render(&mut self, world: &dyn Hittable) {

        let mut imgbuf = image::ImageBuffer::new(self.image_width, self.image_height);
        
        println!("Started generating image of size {} * {}",self.image_width, self.image_height);

        for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
            if x == 0 && y % 10 == 0 {
                println!("({} {})",x,y);
            }

            let result = (0..(self.samples_per_pixel)).map(|_| {
                let ray = self.get_ray(x, y);
                Camera::ray_color(&ray, world)
            }).fold(Vector3::<f32>::default(), |acc, x| acc + x) 
            / (self.samples_per_pixel as f32);

            *pixel = color_to_rgb(result);
        }

        imgbuf.save("image.png").unwrap();

        println!("Finished generating");
    }

    
    fn ray_color(ray: &Ray, hittable: &dyn Hittable) -> Color {
        let mut hit_record = HitRecord::default();

        if hittable.hit(ray, Interval::new(0.0, f32::INFINITY), &mut hit_record) {
            return (hit_record.normal + vector![1.0,1.0,1.0]) / 2.0;
        };

        let unit_direction = ray.direction / ray.direction.norm();
        let a = 0.5 * (unit_direction.y + 1.0);
        vector![1.0,1.0,1.0].lerp(&vector![0.5,0.7,1.0], a)
    } 

    ///Get randomly sample ray from camera at pixel location
    fn get_ray(&mut self, i: u32, j: u32) -> Ray {
        let pixel_center = self.pixel_start + ((i as f32) * self.pixel_delta_u + (j as f32) * self.pixel_delta_v);
        let pixel_sample = pixel_center + self.pixel_sample_square();

        Ray::new(self.center, pixel_sample - self.center)
    }

    //Returns a random point in square surrounding origin of pixel
    fn pixel_sample_square(&mut self) -> Vector3<f32> {
        let px = -0.5 + self.rng.gen::<f32>();
        let py = -0.5 + self.rng.gen::<f32>();
        px * self.pixel_delta_u + py * self.pixel_delta_v
    }
}