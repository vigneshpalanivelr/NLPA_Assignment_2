# Sentiment Analysis Application

A web-based sentiment analysis application that uses Natural Language Processing (NLP) techniques to analyze the sentiment of user-provided text. The application classifies text as positive, negative, or neutral using VADER and TextBlob sentiment analyzers.

## Features

- **Web Interface**: Intuitive UI for text input or file upload
- **Multiple Input Methods**: Direct text entry or upload .txt files
- **Advanced NLP Processing**:
  - Text preprocessing (tokenization, stemming, lemmatization)
  - VADER sentiment analysis (optimized for social media text)
  - TextBlob sentiment analysis (general purpose)
  - Combined analysis for improved accuracy
- **Visual Results**:
  - Color-coded sentiment badges
  - Confidence score visualization
  - Interactive bar charts using Chart.js
  - Detailed score breakdowns
- **Preprocessing Transparency**: View cleaned text, tokens, and processing steps

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **NLTK**: Natural Language Toolkit for text preprocessing
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **TextBlob**: Python library for processing textual data
- **Uvicorn**: ASGI server for running FastAPI

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS Grid and Flexbox
- **Vanilla JavaScript**: No framework dependencies
- **Chart.js**: Data visualization library

## Project Structure

```
sentiment_analysis_app/
├── backend/
│   ├── main.py                  # FastAPI application and API endpoints
│   ├── sentiment_analyzer.py    # Sentiment analysis logic
│   ├── preprocessing.py         # Text preprocessing utilities
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html              # Main HTML page
│   └── static/
│       ├── css/
│       │   └── style.css       # Application styles
│       └── js/
│           └── app.js          # Frontend logic and API integration
└── README.md                   # This file
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

### Step 1: Clone or Navigate to the Project

```bash
cd sentiment_analysis_app/backend
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (web framework and server)
- NLTK and TextBlob (NLP libraries)
- Other required dependencies

### Step 4: Download NLTK Data

The application will automatically download required NLTK data on first run, but you can pre-download it:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('omw-1.4')"
```

## Running the Application

### Method 1: Direct Python Execution

```bash
cd backend
python main.py
```

### Method 2: Using Uvicorn

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

### Accessing the Application

1. Open your web browser
2. Navigate to `http://localhost:8000`
3. The sentiment analysis interface will load

## Usage Instructions

### Method 1: Text Input

1. Select the "Text Input" tab
2. Enter or paste your text in the text area
3. Click "Analyze Text" button
4. View the sentiment analysis results

### Method 2: File Upload

1. Select the "File Upload" tab
2. Click the upload area or drag-and-drop a .txt file
3. Click "Analyze File" button
4. View the sentiment analysis results

### Understanding Results

The application provides comprehensive analysis:

1. **Sentiment Badge**: Color-coded label (Green=Positive, Red=Negative, Gray=Neutral)
2. **Confidence Score**: How confident the model is in its prediction (0-100%)
3. **Sentiment Chart**: Visual bar chart showing score distribution
4. **VADER Analysis**: Compound score and individual sentiment scores
5. **TextBlob Analysis**: Polarity and subjectivity scores
6. **Preprocessing Details**: Shows cleaned text, tokens, and token count

## API Documentation

### Endpoints

#### 1. Health Check
```
GET /api/health
```
Returns API status

#### 2. Analyze Text
```
POST /api/analyze
Content-Type: application/json

{
  "text": "Your text here"
}
```

#### 3. Analyze File
```
POST /api/analyze/file
Content-Type: multipart/form-data

file: <text-file>
```

#### 4. Batch Analysis
```
POST /api/analyze/batch
Content-Type: application/json

["text1", "text2", "text3"]
```

### Interactive API Documentation

FastAPI provides automatic interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## NLP Techniques Used

### 1. Text Preprocessing

- **Tokenization**: Breaking text into individual words
- **Lowercasing**: Normalizing text to lowercase
- **Special Character Removal**: Removing URLs, mentions, special characters
- **Stopword Removal**: Removing common words with little semantic value
- **Lemmatization**: Reducing words to their base form
- **Stemming**: Alternative word reduction technique

### 2. Sentiment Analysis

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Rule-based sentiment analysis
- Optimized for social media text
- Handles emoticons, slang, and intensity modifiers
- Provides compound, positive, negative, and neutral scores

**TextBlob**
- Pattern-based sentiment analysis
- Provides polarity (sentiment) and subjectivity scores
- Good for general text analysis

**Combined Approach**
- Uses both VADER and TextBlob
- VADER weighted more heavily for robustness
- Confidence score based on agreement between methods

## Test Examples

To quickly test all features of the application, use these comprehensive test inputs:

### Test Case 1: Product Review (Mixed Sentiment)
```
Just received my new laptop from https://example.com and WOW!!! 🎉 @TechReviewer you were RIGHT about this! The display is STUNNING and the battery lasts forever 😍 But honestly, the keyboard feels cheap and the touchpad is terrible 😞 Price was $1299 which seems high. Overall it's okay I guess... 7/10 would maybe recommend? #TechReview #Laptop
```
**Tests:** Mixed sentiment, emoticons, URL/mention removal, capitalization, numbers, hashtags

### Test Case 2: Social Media Rant (Sarcasm Detection)
```
OMG this is just PERFECT!!! 🙄 Ordered from www.store.com 3 weeks ago and FINALLY got it today! And guess what? IT'S BROKEN!!! 😡😡😡 Customer service at @BadCompany was sooooo helpful... NOT!!! Spent $500 on this garbage. Yeah, AMAZING quality control guys! 10/10 disappointment! Never shopping here again!!! #Scam #Disappointed #WorstEver
```
**Tests:** Heavy sarcasm, ALL CAPS, multiple emoticons, elongated words (sooooo), fake positive words

### Test Case 3: Detailed Multi-Aspect Review
```
After using this smartphone for 2 months, here's my honest review: The camera quality is EXCEPTIONAL! 📸 Best I've seen under $800. @Samsung really nailed it there! BUT... the battery life is disappointing 😕 barely lasts 6 hours with normal use. Screen is gorgeous (6.5" AMOLED) but it scratches too easily. Performance: 10/10, Design: 8/10, Battery: 3/10, Camera: 9/10. Customer support via https://support.example.com was pretty good tho. Would I recommend? Depends on your priorities I guess 🤷‍♂️ #SmartphoneReview #TechLife #Mixed
```
**Tests:** Multiple aspects, mixed sentiments, technical specs, ratings, informal language

### Test Case 4: Restaurant Review (Emotional + Comparative)
```
Visited Mario's Restaurant last night (www.mariosrest.com) and what a DISASTER!!! 😤 Food was cold, service was TERRIBLE, and they charged us $150!!! The pasta was undercooked, the wine was warm 🍷😒, and don't even get me started on the dessert - looked NOTHING like the photos! @ChefMario seriously?!? My previous visit 2 years ago was amazing but this time... HORRIBLE! The only good thing was the breadsticks lol 😅 Save your money and go to Luigi's instead! Rating: 2/10 ⭐ #FoodReview #Disappointed #NeverAgain
```
**Tests:** Strong negative with one positive element, comparative sentiment, multiple aspects, alternative recommendation

### Test Case 5: Tech Support Experience (Professional + Frustrated)
```
Been trying to fix my router for 3 DAYS!!! 😫 Contacted support at support@techco.com and chatted with @TechSupport247... The first agent was helpful and professional 👍 (thank you Sarah!) but the second one had NO IDEA what they were doing 🤦‍♂️ Spent 2.5 hours on hold listening to terrible music. Finally got it working but ONLY because I found the solution myself on https://forums.techhelp.com/fix-guide!!! Cost me $89 for "premium support" that was basically useless. Mixed feelings: great product, AWFUL support. 6/10 overall. #TechSupport #CustomerService #Frustrating
```
**Tests:** Mixed sentiment (product vs support), emotional progression, email handling, gratitude mixed with frustration

### Test Case 6: Travel Experience (Narrative Style)
```
Just got back from my trip booked via www.travelsite.com 🏖️ The flight with @AirlineXYZ was delayed 4 hours (UGH! 😒) and lost my luggage BUT the hotel was ABSOLUTELY STUNNING!!! 🌟 Beach view was incredible, staff were super friendly, and the food... OMG THE FOOD!!! 😍🍽️ Best seafood ever! However, paid $2000 total and the wifi was TERRIBLE and AC didn't work properly 😓 Would I go again? Maybe, but definitely using a different airline! The destination itself: 10/10 ⭐⭐⭐⭐⭐ Travel experience: 5/10 😐 #Travel #VacationMode #MixedBag
```
**Tests:** Journey narrative, extreme positive and negative in same text, multiple service providers, separate ratings

### Test Case 7: Short But Complex (Contradictory Language)
```
Bought from @Store for $50 😅 It's terribly good! Like, I hate how much I love it??? Quality is awful... awfully AMAZING!!! 🤯 Check it www.example.com 8/10 would confuse again lol #Weird
```
**Tests:** Contradictory language, sarcasm, irony, informal expressions, short but feature-dense

### Quick Test Suite (Simple Examples)

**Strong Positive:**
```
Amazing product! Exceeded my expectations! 😊👍 Highly recommended!
```

**Strong Negative:**
```
Terrible quality. Very disappointed. Waste of money. 😡
```

**Neutral:**
```
The product arrived. It works as described. Standard quality.
```

**Features Tested:**
- ✅ Text preprocessing (tokenization, lemmatization, cleaning)
- ✅ VADER sentiment analysis (handles emoticons, caps, punctuation)
- ✅ TextBlob sentiment analysis (polarity and subjectivity)
- ✅ Combined confidence scoring
- ✅ URL, mention (@), and hashtag (#) removal
- ✅ Word count comparison (original vs processed)
- ✅ Visualization with Chart.js
- ✅ Detailed result breakdown

## Configuration for OSHA Cloud Lab

To run on BITS OSHA Cloud Lab:

1. Upload the entire `sentiment_analysis_app` folder to the lab environment
2. Follow the installation steps above
3. The application is configured to run on `0.0.0.0:8000` to allow external access
4. Access the application using the lab's assigned URL/IP

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Issue**: Port already in use
```bash
# Solution: Use a different port
uvicorn main:app --port 8001
```

**Issue**: NLTK data not found
```bash
# Solution: Download NLTK data manually
python -c "import nltk; nltk.download('all')"
```

**Issue**: Frontend not loading
```bash
# Solution: Ensure you're accessing the root URL (http://localhost:8000)
# Not the /docs or /api endpoints
```

## File Size Limitations

- Maximum file size: 1MB
- Supported file format: .txt only
- UTF-8 encoding required

## Performance Notes

- Text preprocessing is performed in real-time
- Analysis typically completes in < 1 second for normal text
- Batch analysis supports up to 100 texts per request

## Future Enhancements

Possible improvements for the application:
- Support for multiple languages
- Real-time feedback mechanism for model improvement
- Advanced visualization options
- Export results to PDF/CSV
- Historical analysis tracking
- Multi-modal sentiment analysis (text + images)

## Credits

Developed for:
- **Course**: M.Tech. in AIML - NLP Applications
- **Assignment**: Assignment 2 - PS-9
- **Course Code**: S1-25_AIMLCZG519
- **Institution**: BITS Pilani Work Integrated Learning Programmes

## License

This project is developed for educational purposes as part of the BITS Pilani M.Tech. AIML program.
