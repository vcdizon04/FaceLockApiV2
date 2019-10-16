import * as faceapi from 'face-api.js';
import express from 'express';
const port = 4000;
const app = express();
import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons';

const REFERENCE_IMAGE = '../images/bbt1.jpg'
const QUERY_IMAGE = '../images/bbt4.jpg'

const person2url = "vincen2.jpg";
           const person3url = "vincent3.jpg";
           const person4url = "vincent4.jpg";
           const person5url = "vincent5.jpg";



app.post('/recognize', async (req, res) => {
  const referenceImage = await canvas.loadImage(person2url)
  const queryImage = await canvas.loadImage(person5url)

  const resultsRef = await faceapi.detectSingleFace(referenceImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptor()

    console.log(resultsRef);

  const resultsQuery = await faceapi.detectSingleFace(queryImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptor()

  const faceMatcher = new faceapi.FaceMatcher(resultsRef)

  console.log(resultsQuery);

  if (resultsQuery) {
    const bestMatch = faceMatcher.findBestMatch(resultsQuery.descriptor);
    console.log(bestMatch);
     return res.send(bestMatch);

    }
    return res.status(401).send('Error');

});

app.listen(port, async() => {
  await faceDetectionNet.loadFromDisk('weights');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('weights');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('weights');
  console.log(`App listening to port ${port}`);
}
);