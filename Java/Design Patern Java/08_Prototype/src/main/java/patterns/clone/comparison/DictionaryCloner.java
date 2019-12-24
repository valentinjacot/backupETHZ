package patterns.clone.comparison;

import java.lang.reflect.Array;

import com.rits.cloning.Cloner;

@SuppressWarnings("unused")
public class DictionaryCloner implements CloneableDictionary {
	private String language;
	private final int size;
	private final String[] words;

	public DictionaryCloner(String language, int size) {
		this.language = language;
		this.size = size;
		this.words = new String[size];
		for (int i = 0; i < size; i++) {
			this.words[i] = "String " + i;
		}
	}

	@Override
	public DictionaryCloner clone() {
		Cloner c = Cloner.standard();
		// c.setDumpClonedClasses(true);
		c.registerFastCloner(String[].class, (arr, cloner, clones) -> {
			final Class<?> clz = arr.getClass();
			final int length = Array.getLength(arr);
			final String[] newInstance = (String[]) Array.newInstance(clz.getComponentType(), length);
			if (clones != null) {
				clones.put(arr, newInstance);
			}
			System.arraycopy(arr, 0, newInstance, 0, length);
			return newInstance;
		});
		return c.deepClone(this);
	}

}
