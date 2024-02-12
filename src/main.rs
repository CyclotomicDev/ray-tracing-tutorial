use nalgebra::{Vector3, vector};
use std::rc::Rc;


fn main() {
    //Image 
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;

    //Calculate image height: 1) round down 2) ensure at least 1
    let image_height = (image_width as f32/ aspect_ratio) as i32;
    let image_height: u32 = if image_height < 1 {1} else {image_height as u32};

    //World
    let mut world = HitableCollection::new();
    world.push(Rc::new(Sphere::new(vector![0.0,0.0,-1.0], 0.5)));
    world.push(Rc::new(Sphere::new(vector![0.0,-100.5,-1.0], 100.0)));

    //Camera
    let focal_length = 1.0;
    let viewport_height = 2.0;
    let viewport_width = viewport_height * ((image_width as f32) / (image_height as f32));
    let camera_center = vector![0.0,0.0,0.0];

    //Calculate vectors across the horizontal and down vertical viewport edges
    let viewport_u: Point = vector![viewport_width, 0.0, 0.0];
    let viewport_v: Point = vector![0.0, -viewport_height, 0.0];

    //Calculate horizontal and vertical delta (pixel size)
    let pixel_delta_u: Direction = viewport_u / (image_width as f32);
    let pixel_delta_v: Direction = viewport_v / (image_height as f32);

    //Calculate location of upper left pixel
    let viewport_start = 
        camera_center.clone()
        - vector![0.0, 0.0, focal_length]
        - (viewport_u / 2.0)
        - (viewport_v / 2.0);

    let pixel_start =
        viewport_start + 0.5 * (pixel_delta_u + pixel_delta_v);

        

    //Render
    let mut imgbuf = image::ImageBuffer::new(image_width, image_height);
    
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let pixel_center = pixel_start + (x as f32 * pixel_delta_u) + (y as f32 * pixel_delta_v);
        let ray_direction: Direction = pixel_center - camera_center;

        let ray = Ray::new(camera_center, ray_direction);

        *pixel = color_to_rgb(ray_color(&ray,&world));
    }

    imgbuf.save("image.png").unwrap();
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

trait Hitable {
    fn hit(&self, ray: &Ray, t_interval: Interval, rec: &mut HitRecord) -> bool;
}

struct HitableCollection {
    objects: Vec<Rc<dyn Hitable>>,
}

impl HitableCollection {
    fn new() -> Self {
        let objects = Vec::new();
        Self{objects}
    }

    fn push(&mut self, object:  Rc<dyn Hitable>) {
        self.objects.push(object);
    }

    fn clear(&mut self) {
        self.objects.clear();
    }
}

impl Hitable for HitableCollection {
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

impl Hitable for Sphere
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
    let a = 255.999; 
    let color = a * color;
    image::Rgb([color[0] as u8, color[1] as u8, color[2] as u8])
}

fn ray_color(ray: &Ray, hitable: &dyn Hitable) -> Color {
    let mut hit_record = HitRecord::default();
    if hitable.hit(ray, Interval::new(0.0, f32::INFINITY), &mut hit_record) {
        return (hit_record.normal + vector![1.0,1.0,1.0]) / 2.0;
    };

    let unit_direction = ray.direction / ray.direction.norm();
    let a = 0.5 * (unit_direction.y + 1.0);
    vector![1.0,1.0,1.0].lerp(&vector![0.5,0.7,1.0], a)
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
}

const empty: Interval =  Interval{min: f32::INFINITY, max:-f32::INFINITY};
const universal: Interval =  Interval{min:-f32::INFINITY, max:f32::INFINITY};