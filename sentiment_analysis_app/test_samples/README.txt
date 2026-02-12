SENTIMENT ANALYSIS TEST SAMPLES
================================

This folder contains sample text files to test the file upload functionality
of the Sentiment Analysis Application.

Each file tests different aspects of the sentiment analysis system:


FILE DESCRIPTIONS:
-----------------

1. 1_positive_product_review.txt
   - Sentiment: STRONG POSITIVE
   - Features tested:
     * Positive sentiment detection
     * Emoticons (😊⭐)
     * URL removal (www.example.com)
     * Hashtags (#BestPurchase, #HighlyRecommended)
     * Structured review format
     * Multiple positive indicators
     * Ratings (5 stars, 10/10)
   - Expected confidence: HIGH (85-95%)

2. 2_negative_complaint.txt
   - Sentiment: STRONG NEGATIVE
   - Features tested:
     * Negative sentiment detection
     * All caps (WORST, BROKEN, TERRIBLE)
     * Emoticons (😡)
     * Mention handling (@TechStore)
     * URL removal (www.trustpilot.com)
     * Hashtags (#Scam, #Disappointed)
     * Pricing mentions ($299)
     * Multiple negative indicators
     * Ratings (0/10)
   - Expected confidence: HIGH (90-95%)

3. 3_neutral_technical_review.txt
   - Sentiment: NEUTRAL
   - Features tested:
     * Neutral/objective sentiment
     * Technical specifications
     * Numbers and measurements
     * Email address (support@company.com)
     * Factual statements
     * No emotional language
     * Balanced rating (3/5)
   - Expected confidence: MODERATE (70-80%)

4. 4_mixed_smartphone_review.txt
   - Sentiment: MIXED (Neutral or Slightly Positive)
   - Features tested:
     * Mixed positive and negative sentiments
     * Section-based analysis (THE GOOD, THE BAD, THE UGLY)
     * Multiple aspect ratings
     * Emoticons (both positive and negative)
     * Mention handling (@Samsung)
     * URL removal (www.techreview.com)
     * Hashtags (#TechReview, #MixedFeelings)
     * Comparative statements
     * Conditional recommendations
   - Expected confidence: LOWER (60-75%)

5. 5_sarcastic_review.txt
   - Sentiment: NEGATIVE (Despite positive words)
   - Features tested:
     * Sarcasm detection
     * All caps with positive words (AMAZING, OUTSTANDING, PERFECT)
     * Sarcastic emoticons (🙄😒🙃)
     * Contradiction between words and meaning
     * URL removal (www.cheapstore.com, www.regretfulpurchases.com)
     * Mention handling (@CompanyName)
     * Hashtags (#Sarcasm, #WorstEver)
     * VADER's ability to catch sarcasm through punctuation
   - Expected confidence: MODERATE-HIGH (75-85%)
   - NOTE: VADER typically catches sarcasm better due to caps and punctuation

6. 6_social_media_style.txt
   - Sentiment: STRONG POSITIVE
   - Features tested:
     * Social media informal style
     * Heavy emoticon usage (🎉🔥😍💯)
     * Repeated punctuation (!!!)
     * Elongated words (INSANE, SCREAMING)
     * Multiple mentions (@AudioTech, @MusicProducer, @DJLife)
     * URL removal (www.audiotech.com)
     * Many hashtags (#Headphones, #AudioTech, etc.)
     * Informal expressions (OMG, lol, tbh, yaasss)
     * Pros/cons list format
     * Mixed with slight criticism but overall very positive
   - Expected confidence: HIGH (85-95%)


HOW TO USE THESE FILES:
-----------------------

1. Start the Sentiment Analysis Application
2. Navigate to the "File Upload" tab
3. Click the upload area or drag and drop any of these .txt files
4. Click "Analyze File" button
5. Review the results:
   - Check if sentiment classification matches expected sentiment
   - Verify confidence score is in expected range
   - Review preprocessing details:
     * Original word count
     * Token count after preprocessing
     * Words removed percentage
   - Examine VADER and TextBlob scores
   - View sentiment distribution chart
   - Check that URLs, mentions, and hashtags were removed


WHAT TO OBSERVE:
----------------

Preprocessing:
- Original Word Count: How many words were in the file
- Token Count: Words remaining after removing stopwords, cleaning
- Words Removed: Shows the impact of preprocessing

Sentiment Scores:
- VADER: Better for social media, sarcasm, emoticons
- TextBlob: Better for formal, structured text
- Combined: Final verdict based on both models

Confidence:
- HIGH confidence (>80%): Clear, consistent sentiment across text
- MODERATE confidence (60-80%): Mixed or ambiguous sentiment
- LOWER confidence (<60%): Contradictory or highly mixed content


EXPECTED PREPROCESSING IMPACT:
------------------------------

File 1: ~50-60% word reduction (many stopwords in formal review)
File 2: ~45-55% word reduction (emotional language, less stopwords)
File 3: ~40-50% word reduction (technical terms preserved)
File 4: ~50-60% word reduction (long detailed review)
File 5: ~45-55% word reduction (sarcastic but has structure)
File 6: ~55-65% word reduction (social media has many fillers, emoticons)


TROUBLESHOOTING:
---------------

If you encounter errors:
- Ensure file is .txt format (not .doc, .docx, .pdf)
- Check file size is under 1MB
- Verify file encoding is UTF-8
- Make sure backend server is running on http://localhost:8000


NOTES:
------

- These files are designed to test ALL features of the application
- They contain intentional variations in:
  * Sentiment (positive, negative, neutral, mixed, sarcastic)
  * Length (short to long)
  * Style (formal, informal, social media, technical)
  * Special characters (emoticons, URLs, mentions, hashtags)
  * Structure (lists, sections, ratings)

- Use these to verify the application handles real-world text correctly

- Feel free to modify these files or create your own test cases!


Happy Testing! 🚀
