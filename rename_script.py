import os
import glob

files = [f for f in glob.glob('d:/DL_proj/**/*', recursive=True) if os.path.isfile(f) and f.endswith(('.py', '.md', '.yaml', '.txt'))]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            
        if 'tweet_sentiment_classifier' in content:
            new_content = content.replace('tweet_sentiment_classifier', 'tweet_sentiment_classifier')
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)
    except Exception as e:
        print(f"Skipping {f}: {e}")
