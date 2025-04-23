import unittest
from flask import Flask
from app import app

class EmotionDetectionAppTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        # Test the home page
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to the Live Emotion Detection System', response.data)

    def test_page1(self):
        # Test the FER-2013 model page
        response = self.app.get('/page1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Emotion Detection - FER Model', response.data)

    def test_page2(self):
        # Test the custom emotion model page
        response = self.app.get('/page2')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Emotion Detection - My Emotion Model', response.data)

    def test_video_feed_fer(self):
        # Test the video feed for FER model
        response = self.app.get('/video_feed_fer')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)

    def test_video_feed_emotion(self):
        # Test the video feed for custom emotion model
        response = self.app.get('/video_feed_emotion')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)

    def test_change_camera(self):
        # Test changing the camera
        response = self.app.get('/change_camera/1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Camera changed successfully', response.data)

    def test_invalid_route(self):
        # Test an invalid route
        response = self.app.get('/invalid_route')
        self.assertEqual(response.status_code, 404)

    def test_video_feed_with_invalid_camera(self):
        # Test the video feed with an invalid camera index
        response = self.app.get('/video_feed_fer?camera=999')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)

    def test_training_graph(self):
        # Test the training graph route
        response = self.app.get('/training_graph')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'image/png')

    def test_camera_change_persistence(self):
        # Test if the camera change persists across requests
        self.app.get('/change_camera/1')
        response = self.app.get('/video_feed_fer')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)

    def test_home_page_content(self):
        # Test the content of the home page
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Choose Your Options', response.data)
        self.assertIn(b'FER-2013 Model', response.data)
        self.assertIn(b'My Emotions Model', response.data)

    def test_page1_content(self):
        # Test the content of the FER-2013 model page
        response = self.app.get('/page1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Emotion Detection - FER Model', response.data)
        self.assertIn(b'Select Camera', response.data)
        self.assertIn(b'Live Video Feed', response.data)

    def test_page2_content(self):
        # Test the content of the custom emotion model page
        response = self.app.get('/page2')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Emotion Detection - My Emotion Model', response.data)
        self.assertIn(b'Select Camera', response.data)
        self.assertIn(b'Live Video Feed', response.data)

    def test_real_time_face_emotion_detection(self):
        # Test real-time face emotion detection
        response = self.app.get('/video_feed_fer')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)
        # Additional checks can be added to verify the content of the frames

    def test_emotion_mapping(self):
        # Test emotion mapping
        response = self.app.get('/video_feed_emotion')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)
        # Additional checks can be added to verify the content of the frames

    def test_error_handling(self):
        # Test error handling for missing model
        response = self.app.get('/video_feed_fer?camera=999')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'--frame', response.data)
        # Additional checks can be added to verify error messages

if __name__ == '__main__':
    unittest.main()
