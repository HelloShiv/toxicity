import { NextRequest, NextResponse } from 'next/server';
import { load } from '@tensorflow-models/toxicity';  // Corrected import for 'load'

// The minimum prediction confidence.
const threshold = 0.9;

// Load the toxicity model once
let model = null;

async function loadModel() {
  if (!model) {
    model = await load(threshold);  // Load the model with the confidence threshold
    console.log('Model loaded');
  }
  return model;
}

// API route to process the comment and return toxicity predictions
export async function POST(req) {
  console.log('Received request');
  const { comment } = await req.json();

  console.log('Received comment:', comment);

  if (!comment || typeof comment !== 'string') {
    return NextResponse.json({ error: 'Comment text is required and must be a string' }, { status: 400 });
  }

  try {
    // Load the model if it's not loaded already
    const loadedModel = await loadModel();
    
    // Get predictions from the model
    const predictions = await loadedModel.classify([comment]);

    // Format the predictions to send back in a readable way
    const result = predictions.map(prediction => ({
      label: prediction.label,
      results: prediction.results.map(result => ({
        match: result.match,
        probabilities: result.probabilities,
      })),
    }));

    return NextResponse.json(result);
  } catch (error) {
    console.error('Error during classification:', error);
    return NextResponse.json({ error: 'Error processing the comment' }, { status: 500 });
  }
}
