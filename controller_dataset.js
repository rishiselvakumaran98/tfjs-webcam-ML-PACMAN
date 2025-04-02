/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
    this.examplesByLabel = Array.from({ length: numClasses }, () => []);
    this.xs = null;
    this.ys = null;
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} example A tensor representing the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be a number.
   */
  addExample(example, label) {
    // One-hot encode the label.
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    this.examplesByLabel[label].push(example);

    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // ControllerDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }

  updateTensors() {
    const xsList = [];
    const ysList = [];
    for (let label = 0; label < this.numClasses; label++) {
      const examples = this.examplesByLabel[label];
      for (let i = 0; i < examples.length; i++) {
        xsList.push(examples[i]);
        // Create one-hot encoding for the label.
        const oneHot = tf.tidy(() =>
          tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));
        ysList.push(oneHot);
      }
    }

    // Dispose previous xs and ys.
    if (this.xs) this.xs.dispose();
    if (this.ys) this.ys.dispose();

    if (xsList.length > 0) {
      // Concatenate all examples.
      this.xs = tf.keep(tf.concat(xsList, 0));
      this.ys = tf.keep(tf.concat(ysList, 0));
    } else {
      this.xs = null;
      this.ys = null;
    }

    // Dispose temporary one-hot tensors.
    ysList.forEach(t => t.dispose());
  }

  clearExamplesForLabel(label) {
    // Dispose each example for this label.
    this.examplesByLabel[label].forEach(example => example.dispose());
    // Reset the array for that label.
    this.examplesByLabel[label] = [];
    // Rebuild xs and ys based on remaining examples.
    this.updateTensors();
  }
}
