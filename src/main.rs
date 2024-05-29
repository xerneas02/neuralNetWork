#![allow(dead_code)]

extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate mnist;

use image::{GrayImage, Luma};
use ndarray::{s, Array2, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use mnist::{Mnist, MnistBuilder};
use serde::{Serialize,Deserialize};
use std::error::Error;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
}

impl NeuralNetwork {
    fn save(&self, filename: &str) -> io::Result<()> {
        let serialized = serde_json::to_string(&self).unwrap();
        let mut file = File::create(filename)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    fn load(filename: &str) -> io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let nn: NeuralNetwork = serde_json::from_str(&contents).unwrap();
        Ok(nn)
    }

    fn new_fixed_sizes(input_size: usize, hidden_size: usize, num_hidden_layers: usize, output_size: usize) -> Self {
        let hidden_sizes = vec![hidden_size; num_hidden_layers];
        Self::new(input_size, &hidden_sizes, output_size)
    }

    fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut weights = Vec::with_capacity(hidden_sizes.len() + 1);
        let mut biases = Vec::with_capacity(hidden_sizes.len() + 1);

        weights.push(Array2::<f64>::random((input_size, hidden_sizes[0]), Uniform::new(-1.0, 1.0)));
        biases.push(Array2::<f64>::random((1, hidden_sizes[0]), Uniform::new(-1.0, 1.0)));

        for i in 1..hidden_sizes.len() {
            weights.push(Array2::<f64>::random((hidden_sizes[i - 1], hidden_sizes[i]), Uniform::new(-1.0, 1.0)));
            biases.push(Array2::<f64>::random((1, hidden_sizes[i]), Uniform::new(-1.0, 1.0)));
        }

        weights.push(Array2::<f64>::random((hidden_sizes[hidden_sizes.len() - 1], output_size), Uniform::new(-1.0, 1.0)));
        biases.push(Array2::<f64>::random((1, output_size), Uniform::new(-1.0, 1.0)));

        NeuralNetwork { weights, biases }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        let sigmoid = NeuralNetwork::sigmoid(x);
        sigmoid * (1.0 - sigmoid)
    }

    fn forward(&self, mut inputs: Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut activations = Vec::with_capacity(self.weights.len() + 1);
        let mut zs = Vec::with_capacity(self.weights.len());

        activations.push(inputs.clone());

        for (weights, biases) in self.weights.iter().zip(&self.biases) {
            let z = inputs.dot(weights) + biases;
            let activation = z.mapv(NeuralNetwork::sigmoid);
            zs.push(z);
            activations.push(activation.clone());
            inputs = activation;
        }

        (activations, zs)
    }

    fn cost(&self, output: &Array2<f64>, expected: &Array2<f64>) -> f64 {
        let diff = output - expected;
        let squared_diff = diff.mapv(|x| x.powi(2));
        squared_diff.mean().unwrap()
    }

    fn train(&mut self, inputs: Array2<f64>, expected: Array2<f64>, learning_rate: f64, epochs: usize, batch_size: usize) {
        let num_batches = inputs.len_of(Axis(0)) / batch_size;

        for epoch in 0..epochs {
            let mut total_cost = 0.0;

            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = start_idx + batch_size;
                let input_batch = inputs.slice(s![start_idx..end_idx, ..]).to_owned();
                let expected_batch = expected.slice(s![start_idx..end_idx, ..]).to_owned();

                let (activations, zs) = self.forward(input_batch.clone());
                let outputs = activations.last().unwrap().clone();
                let cost = self.cost(&outputs, &expected_batch);
                total_cost += cost;

                // Backpropagation
                let mut delta = &outputs - &expected_batch;
                delta = delta * zs.last().unwrap().mapv(NeuralNetwork::sigmoid_derivative);

                for i in (0..self.weights.len()).rev() {
                    let a_prev = activations[i].clone();

                    let gradient_weights = a_prev.t().dot(&delta) * learning_rate / batch_size as f64;
                    let gradient_biases = delta.sum_axis(Axis(0)) * learning_rate / batch_size as f64;

                    self.weights[i] -= &gradient_weights;
                    self.biases[i] -= &gradient_biases;

                    if i > 0 {
                        delta = delta.dot(&self.weights[i].t());
                        let z = zs[i - 1].clone();
                        delta = delta * z.mapv(NeuralNetwork::sigmoid_derivative);
                    }
                }
            }

            if epoch%100 == 0 {println!("Epoch {}: Average Cost = {:?}", epoch + 1, total_cost / num_batches as f64);}
        }
    }
}


fn print_array(array : ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>) {
    for i in 0..array.len() {
        print!("{} ", array.get(i).unwrap());
    }
}

fn train(mnist_path: &str, model_path: &str) {
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .base_path(mnist_path)
        .label_format_one_hot()
        .finalize();

    let images = Array2::from_shape_vec((60_000, 28 * 28), trn_img).unwrap().mapv(|x| x as f64 / 255.0);

    let labels = Array2::from_shape_vec((60_000, 10), trn_lbl).unwrap().mapv(|x| x as f64);

    let nn_result = NeuralNetwork::load(model_path);

    let mut nn = match nn_result {
        Ok(nn) => nn,
        Err(err) => {
            eprintln!("Error loading neural network: {}", err);
            NeuralNetwork::new_fixed_sizes(28 * 28, 64, 16, 10)
        }
    };
    
    let batch_size = 128;
    let epochs = 1000;


    nn.train(images, labels, 0.01, epochs, batch_size);

    let test_images = Array2::from_shape_vec((10_000, 28 * 28), tst_img).unwrap().mapv(|x| x as f64 / 255.0);
    let test_labels = Array2::from_shape_vec((10_000, 10), tst_lbl).unwrap().mapv(|x| x as f64);

    let (test_activations, _) = nn.forward(test_images.clone());
    let test_outputs = test_activations.last().unwrap().clone();
    let test_cost = nn.cost(&test_outputs, &test_labels);
    println!("Test Cost: {:?}", test_cost);

    let testing_size = 1000;
    let mut acuracy = 0.0;

    for i in 0..testing_size {
        let image = test_images.slice(s![i..i+1, ..]).to_owned();
        let (pred_activations, _) = nn.forward(image);
        let prediction = pred_activations.last().unwrap().clone();
        
        let predict = max_index(&prediction).unwrap().1;

        if *(test_labels.slice(s![i..i+1, ..]).row(0).get(predict).unwrap()) == 1.0 {
            acuracy += 1.0;
        }
    }

    print!("Acuracy: {}\n", acuracy/testing_size as f64);
    nn.save(model_path).unwrap();
}

fn vec_to_array2(vec: Vec<f64>, rows: usize, cols: usize) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ndarray::ShapeError> {
    Array2::from_shape_vec((rows, cols), vec)
}

fn array_to_image(array: &ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>, image_path: &str) -> Result<(), Box<dyn Error>> {
    // Ensure the input array has the correct length
    if array.len() != 28 * 28 {
        return Err("The input array must have 784 elements.".into());
    }

    // Create a new 28x28 grayscale image
    let mut img: GrayImage = GrayImage::new(28, 28);

    // Populate the image with pixel values from the array
    for (i, &value) in array.iter().enumerate() {
        let x = (i % 28) as u32;
        let y = (i / 28) as u32;
        // Convert the value back to a grayscale pixel (0.0 to 1.0 -> 0 to 255)
        let pixel_value = ((1.0 - value) * 255.0).round() as u8;
        img.put_pixel(x, y, Luma([pixel_value]));
    }

    // Save the image to the specified path
    img.save(Path::new(image_path))?;

    Ok(())
}

fn load_image_as_array(image_path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Load the image
    let img = image::open(image_path)?;

    // Ensure the image is grayscale
    let gray_img = img.to_luma8();

    // Resize the image to 28x28 if necessary
    let resized_img = image::imageops::resize(&gray_img, 28, 28, image::imageops::FilterType::Nearest);

    // Convert the image to an array of size 784 with values between 0.0 and 1.0
    let mut image_array = Vec::with_capacity(28 * 28);
    for pixel in resized_img.pixels() {
        let value =  1.0 - (pixel[0] as f64 / 255.0);
        image_array.push(value);
    }

    Ok(image_array)
}

fn max_index(array: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>) -> Option<(usize, usize)> {
    if array.is_empty() {
        return None; // Return None if the array is empty
    }

    let mut max_index = (0, 0);
    let mut max_value = array[(0, 0)];

    for ((i, j), &value) in array.indexed_iter() {
        if value > max_value {
            max_value = value;
            max_index = (i, j);
        }
    }

    Some(max_index)
}

fn main() {
    //train("D:\\Programation\\rust\\neuralNetWork\\data", "neural_network.json");

    let nn = NeuralNetwork::load("neural_network.json").unwrap();
    
    let image_path = "number.png";
    let image_array = vec_to_array2(load_image_as_array(image_path).unwrap(), 1, 784).unwrap();

    let (activations, _) = nn.forward(image_array);

    println!("{:?}", activations.last().unwrap());
}
