package patterns.clone.comparison;

import java.io.Serializable;

import patterns.clone.comparison.util.SerializableClone;

@SuppressWarnings("unused")
public class DictionarySerialization implements Serializable, CloneableDictionary {
	private static final long serialVersionUID = 6503996804174031728L;

	private String language;
	private final int size;
	private String[] words;

	public DictionarySerialization(String language, int size) {
		this.language = language;
		this.size = size;
		this.words = new String[size];
		for (int i = 0; i < size; i++) {
			this.words[i] = "String " + i;
		}
	}

	@Override
	public DictionarySerialization clone() {
		return (DictionarySerialization)SerializableClone.clone(this);
	}

}
