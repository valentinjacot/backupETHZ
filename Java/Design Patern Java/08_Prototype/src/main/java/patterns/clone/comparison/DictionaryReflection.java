package patterns.clone.comparison;

import patterns.clone.comparison.util.ReflectiveClone;

@SuppressWarnings("unused")
public class DictionaryReflection implements CloneableDictionary {
	private String language;
	private /* final */ int size;
	private String[] words;

	public DictionaryReflection(String language, int size) {
		this.language = language;
		this.size = size;
		this.words = new String[size];
		for (int i = 0; i < size; i++) {
			this.words[i] = "String " + i;
		}
	}

	private DictionaryReflection() {
	}

	@Override
	public DictionaryReflection clone() {
		return (DictionaryReflection)ReflectiveClone.clone(this);
	}
}
