export interface SampleMedia {
  id: string;
  title: string;
  type: 'video' | 'image';
  groundTruth: 'REAL' | 'DEEPFAKE';
  format: 'MP4' | 'MOV' | 'AVI' | 'JPG' | 'PNG';
  sizeFormatted: string;
  sourceDataset: string;
  manipulationType?: string;
  description: string;
  previewUrl: string; // Video or image URL
  simulatedPreset: {
    duration: number;
    recommendedFrames: number;
    expectedPrediction: 'REAL' | 'DEEPFAKE';
    expectedConfidence: number;
    fakeFrameRatio: number;
    artifacts: string[];
  };
}

export const SAMPLE_MEDIA_LIST: SampleMedia[] = [
  {
    id: 'sample-df-video-1',
    title: 'FaceSwap Manipulation (FF++ Deepfakes)',
    type: 'video',
    groundTruth: 'DEEPFAKE',
    format: 'MP4',
    sizeFormatted: '14.2 MB',
    sourceDataset: 'FaceForensics++ (c23 compression)',
    manipulationType: 'AutoEncoder FaceSwap + Boundary Feathering',
    description: 'Target actor with swapped identity face. Displays unnatural blending seams around the jawline, irregular blink rates, and micro-tremors in facial landmarks.',
    previewUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
    simulatedPreset: {
      duration: 15,
      recommendedFrames: 120,
      expectedPrediction: 'DEEPFAKE',
      expectedConfidence: 93.4,
      fakeFrameRatio: 0.85, // 85% fake, 15% real
      artifacts: [
        'Jawline blending boundary discontinuity detected',
        'Abnormal temporal eye blinking cadence',
        'High-frequency residual spectral noise in mouth region',
        'Corneal specular reflection asymmetry'
      ]
    }
  },
  {
    id: 'sample-real-video-1',
    title: 'Authentic Studio Interview (Pristine Video)',
    type: 'video',
    groundTruth: 'REAL',
    format: 'MP4',
    sizeFormatted: '18.6 MB',
    sourceDataset: 'YouTube Pristine HD Dataset (Original ID #042)',
    manipulationType: 'None (Pristine Unmanipulated)',
    description: 'Natural human subject with organic skin texture, physiologically coherent blood-flow chromatic changes, and consistent temporal 3D head rotation.',
    previewUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    simulatedPreset: {
      duration: 12,
      recommendedFrames: 100,
      expectedPrediction: 'REAL',
      expectedConfidence: 96.8,
      fakeFrameRatio: 0.04, // 4% fake, 96% real
      artifacts: [
        'Consistent biological micro-texture and skin pores verified',
        'Physiological corneal reflection symmetry intact',
        'Smooth landmark velocity across temporal timeline'
      ]
    }
  },
  {
    id: 'sample-df-image-1',
    title: 'Synthetic Diffusion Face (StyleGAN3 / Midjourney)',
    type: 'image',
    groundTruth: 'DEEPFAKE',
    format: 'PNG',
    sizeFormatted: '2.4 MB',
    sourceDataset: 'CelebA-HQ Synthetic vs Real Challenge',
    manipulationType: 'Generative Latent Diffusion (Synthetic Face)',
    description: 'High-definition synthetic portrait showing typical GAN/Diffusion background warping, iris pupil edge irregularities, and mismatched earring symmetry.',
    previewUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=600&q=80',
    simulatedPreset: {
      duration: 0,
      recommendedFrames: 1,
      expectedPrediction: 'DEEPFAKE',
      expectedConfidence: 94.7,
      fakeFrameRatio: 1.0,
      artifacts: [
        'Iris border elliptical eccentricity anomaly',
        'Hair strand continuity breakdown against background',
        'Mismatched ear helix topology'
      ]
    }
  },
  {
    id: 'sample-real-image-1',
    title: 'Authentic DSLR Portrait (Natural Lighting)',
    type: 'image',
    groundTruth: 'REAL',
    format: 'JPG',
    sizeFormatted: '1.8 MB',
    sourceDataset: 'Flickr-Faces-HQ (Pristine Camera Raw)',
    manipulationType: 'None (Natural Optical Sensor Capture)',
    description: 'Authentic camera capture with natural depth-of-field, realistic subsurface skin scattering, and continuous specular lighting reflections.',
    previewUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=600&q=80',
    simulatedPreset: {
      duration: 0,
      recommendedFrames: 1,
      expectedPrediction: 'REAL',
      expectedConfidence: 97.2,
      fakeFrameRatio: 0.0,
      artifacts: [
        'Natural sensor noise distribution (PRNU signature intact)',
        'Physiologically correct facial depth contours',
        'Natural eyelid folds and skin micro-creases'
      ]
    }
  },
  {
    id: 'sample-collage-1',
    title: 'Real vs AI-Generated Costume (Side-by-Side Collage)',
    type: 'image',
    groundTruth: 'DEEPFAKE',
    format: 'JPG',
    sizeFormatted: '1.2 MB',
    sourceDataset: 'Synthetic Diffusion vs Authentic Camera Test Suite',
    manipulationType: 'Dual Collage (Left = Real Photo, Right = AI Generated Armor)',
    description: 'Split-photo evidence benchmark. The Left photo is authentic camera photography, while the Right photo is AI-generated with fantasy armor texturing and diffusion skin smoothing.',
    previewUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=600&q=80',
    simulatedPreset: {
      duration: 0,
      recommendedFrames: 1,
      expectedPrediction: 'DEEPFAKE',
      expectedConfidence: 98.6,
      fakeFrameRatio: 0.5,
      artifacts: [
        'Direct Comparison: The RIGHT photo is AI Generated (Fake); the LEFT photo is Authentic (Real)',
        'Right photo: Synthetic fantasy armor diffusion texturing and waxy porcelain skin smoothing',
        'Left photo: Authentic optical sensor noise and genuine textile sweater weave',
        'Right photo: Latent upsampler lattice noise and diffuse spectacle reflections'
      ]
    }
  }
];
