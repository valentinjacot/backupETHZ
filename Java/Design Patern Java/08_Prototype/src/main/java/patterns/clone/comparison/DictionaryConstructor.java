package patterns.clone.comparison;

public class DictionaryConstructor implements CloneableDictionary {
	private String language;
	private final int size;
	private String[] words;

	public DictionaryConstructor(String language, int size) {
		this.language = language;
		this.size = size;
		this.words = new String[size];
		for (int i = 0; i < size; i++) {
			this.words[i] = "String "+i;
		}
	}

	public DictionaryConstructor(DictionaryConstructor orig) {
		this.language = orig.language;
		this.size = orig.size;
		if (orig.words != null) {
			this.words = orig.words.clone();
			
//			this.words = new String[size];
//			for (int i = 0; i < size; i++) {
//				this.words[i] = orig.words[i];
//			}
//			
//			this.words = new String[size];
//			System.arraycopy(orig.words, 0, this.words, 0, size);
		}
	}

	@Override
	public DictionaryConstructor clone() {
		return new DictionaryConstructor(this);
	}
}
