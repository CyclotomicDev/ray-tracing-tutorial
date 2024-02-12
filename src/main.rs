fn main() {
    //Image 
    let image_width = 256;
    let image_height = 256;

    //Render
    let mut imgbuf = image::ImageBuffer::new(image_width, image_width);

    let a = 255.999; 
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let r = (x as f32) / (image_width - 1) as f32;
        let g = (y as f32) / (image_height - 1) as f32;
        let b = 0.0;

        let r = (a * r as f32) as u8;
        let g = (a * g as f32) as u8;
        let b = (a * b as f32) as u8;

        *pixel = image::Rgb([r,g,b]);
    }

    imgbuf.save("image.png").unwrap();
}
