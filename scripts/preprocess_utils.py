"""
Text Preprocessing Utilities for HinglishSarc
Handles Hinglish text normalization, cleaning, and preprocessing
"""

import re
import string
import unicodedata
import emoji


class HinglishPreprocessor:
    """Preprocessor for Hinglish (Hindi-English code-mixed) text"""
    
    def __init__(self, 
                 lowercase=True,
                 remove_urls=True,
                 remove_mentions=True,
                 remove_hashtags=False,
                 normalize_whitespace=True,
                 preserve_emojis=True,
                 remove_punctuation=False):
        """
        Initialize preprocessor with configuration
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            normalize_whitespace: Normalize multiple spaces to single space
            preserve_emojis: Keep emojis in text
            remove_punctuation: Remove punctuation marks
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.normalize_whitespace = normalize_whitespace
        self.preserve_emojis = preserve_emojis
        self.remove_punctuation = remove_punctuation
        
        # Common Hinglish transliteration variations
        self.transliteration_map = {
            'ki': ['k', 'ke', 'ki'],
            'hai': ['h', 'hai', 'he'],
            'toh': ['to', 'toh'],
            'kar': ['kr', 'kar'],
            'aur': ['or', 'aur'],
            'nahi': ['nhi', 'nahi', 'nahin'],
        }
    
    def remove_urls_func(self, text):
        """Remove URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        # Also remove www.example.com style
        text = re.sub(r'www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}', '', text)
        return text
    
    def remove_mentions_func(self, text):
        """Remove @mentions"""
        return re.sub(r'@[\w]+', '', text)
    
    def remove_hashtags_func(self, text):
        """Remove #hashtags but keep the text"""
        # Keep the word, just remove the #
        return re.sub(r'#', '', text)
    
    def normalize_whitespace_func(self, text):
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_punctuation_func(self, text):
        """Remove punctuation but preserve sentence structure"""
        # Keep sentence-ending punctuation for sentence splitting
        punctuation = string.punctuation.replace('.', '').replace('!', '').replace('?', '')
        translator = str.maketrans(punctuation, ' ' * len(punctuation))
        return text.translate(translator)
    
    def handle_emojis(self, text):
        """Convert emojis to text descriptions or remove them"""
        if self.preserve_emojis:
            # Keep emojis as-is
            return text
        else:
            # Remove emojis
            return emoji.replace_emoji(text, replace='')
    
    def normalize_unicode(self, text):
        """Normalize unicode characters"""
        # Normalize to NFKC form (compatibility composition)
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def clean_text(self, text):
        """Apply all cleaning steps"""
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.remove_urls_func(text)
        
        # Remove mentions
        if self.remove_mentions:
            text = self.remove_mentions_func(text)
        
        # Handle hashtags
        if self.remove_hashtags:
            text = self.remove_hashtags_func(text)
        
        # Handle emojis
        text = self.handle_emojis(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self.remove_punctuation_func(text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.normalize_whitespace_func(text)
        
        return text
    
    def preprocess(self, text):
        """Main preprocessing function"""
        return self.clean_text(text)
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]


def split_into_sentences(text):
    """
    Split text into sentences for intra-text trajectory modeling
    
    Args:
        text: Input text string
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting on common punctuation
    # This handles both English and Hinglish
    sentences = re.split(r'[.!?।]+', text)
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# Example usage
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = HinglishPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        preserve_emojis=True
    )
    
    test_texts = [
        "Yaar mere tu Aise rutha jaise mera rab rutha",
        "@ViratBharat Raza Murad bahut kaamyab aadmi he. Uski baat sunni chahiye :P #sarcasm",
        "Check this http://example.com for more info! 😂😂"
    ]
    
    print("Testing Hinglish Preprocessor:\n")
    for text in test_texts:
        cleaned = preprocessor.preprocess(text)
        sentences = split_into_sentences(cleaned)
        print(f"Original:  {text}")
        print(f"Cleaned:   {cleaned}")
        print(f"Sentences: {sentences}")
        print("-" * 80)
