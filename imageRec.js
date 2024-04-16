import * as tf from '@tensorflow/tfjs';
// import { loadImage, imageToTensor } from './image-utils';

// Завантаження зображення
const image = loadImage('path/to/your/image.jpg');

// Перетворення зображення у тензор
const tensor = imageToTensor(image);

// Нормалізація тензору
const normalizedTensor = tensor.div(255).expandDims(0);

// Подання тензору на модель
const prediction = model.predict(normalizedTensor);

// Обробка результатів
prediction.print();
