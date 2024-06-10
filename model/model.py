import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

 nltk.download('punkt')
    nltk.download('stopwords')
    data = {
    'ratings': [5, 3, 4, 2, 5, 1, 1, 1, 2, 2, 3, 4, 5, 4, 5, 2, 1, 3, 4, 3, 4, 5, 2, 1, 4, 3, 5, 5, 2, 4],
    'comments': [
        "Great event, very well organized!",
        "It was okay, but could have been better.",
        "Good event, I enjoyed it.",
        "Not satisfied, the event was poorly managed.",
        "Excellent event, would recommend to others!",
        "Terrible event, everything went wrong.",
        "Worst event ever, very disappointing.",
        "The event was a complete disaster, poorly organized and not enjoyable at all.",
        "Not happy with the event, it could have been much better.",
        "Bad organization, did not enjoy the event.",
        "Mediocre event, needs improvement.",
        "Satisfactory event, had a good time.",
        "Fantastic event, highly recommend!",
        "Well organized and fun!",
        "Superb experience, loved every moment.",
        "Disorganized and confusing.",
        "Very disappointing, not what I expected.",
        "Pretty good, but there's room for improvement.",
        "Enjoyed the event, it was fun!",
        "Average event, nothing special.",
        "Good effort, but can be better.",
        "Outstanding event, very memorable.",
        "Needs better management.",
        "Terrible, wouldn't attend again.",
        "Great activities, poorly executed.",
        "Fun event, well done!",
        "Loved the event, fantastic job!",
        "Excellent planning and execution.",
        "Could be improved in many areas.",
        "Good event, but had some issues."
    ],
    'recommendations': [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
    'event_quality': [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]
}


    df = pd.DataFrame(data)

 
    df['comments'] = df['comments'].apply(preprocess_text)

 
    vectorizer = TfidfVectorizer(max_features=100)
    X_text = vectorizer.fit_transform(df['comments']).toarray()

 
    X = np.hstack((df[['ratings', 'recommendations']].values, X_text))
    y = df['event_quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))
    def analyze_real_time_feedback(rating, comment, recommendation):
        comment = preprocess_text(comment)
        comment_vector = vectorizer.transform([comment]).toarray()
        features = np.hstack((np.array([[rating, recommendation]]), comment_vector))
        prediction = model.predict(features)
        return 'Good Quality' if prediction[0] == 1 else 'Poor Quality'
    print(rating)
    print()
    new_feedback = {
        'rating': rating,
        'comment': description,
        'recommendation': recommendation
    }
    insight = analyze_real_time_feedback(new_feedback['rating'], new_feedback['comment'], new_feedback['recommendation'])

    with sqlite3.connect('config.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO feedback (rating, comment, recommendation, insights) VALUES (?, ?, ?, ?)",
                  (rating, description, recommendation, insight))
        conn.commit()

    print(f'Feedback Insight: {insight}')
    return jsonify(message=insight)