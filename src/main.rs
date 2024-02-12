use nalgebra::{Vector3, vector};

fn main() {
    //Image 
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;

    //Calculate image height: 1) round down 2) ensure at least 1
    let image_height = (image_width as f32/ aspect_ratio) as i32;
    let image_height: u32 = if image_height < 1 {1} else {image_height as u32};

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

    let sphere = Sphere::new(vector![0.0,0.0,-1.0], 0.5);
    
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let pixel_center = pixel_start + (x as f32 * pixel_delta_u) + (y as f32 * pixel_delta_v);
        let ray_direction: Direction = pixel_center - camera_center;

        let ray = Ray::new(camera_center, ray_direction);

        *pixel = color_to_rgb(ray_color(&ray,&sphere));
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

    fn _at(&self, t: f32) -> Point {
        self.origin + (t * self.direction)
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

    fn intersection(&self, ray: &Ray) -> bool {
        let dif = ray.origin.clone() - self.center;
        let a = ray.direction.norm_squared();
        let b = 2.0 * ray.direction.dot(&dif);
        let c = dif.norm_squared() - self.radius * self.radius;

        b * b - 4.0 * a * c >= 0.0
    }
}

fn color_to_rgb(color: Color) -> image::Rgb<u8> {
    let a = 255.999; 
    let color = a * color;
    image::Rgb([color[0] as u8, color[1] as u8, color[2] as u8])
}

fn ray_color(ray: &Ray, sphere: &Sphere) -> Color {
    if sphere.intersection(ray) {
        return vector![1.0,0.0,0.0];
    }
    let unit_direction = ray.direction / ray.direction.norm();
    let a = 0.5 * (unit_direction.y + 1.0);
    vector![1.0,1.0,1.0].lerp(&vector![0.5,0.7,1.0], a)
}