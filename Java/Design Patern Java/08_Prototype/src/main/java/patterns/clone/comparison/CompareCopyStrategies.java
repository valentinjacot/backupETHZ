package patterns.clone.comparison;

import java.util.LinkedHashMap;

public class CompareCopyStrategies {

	public static void main(String[] args) {
		final int SIZE = 10000;
		final int NOFCLONES = 1000;
		
		LinkedHashMap<String, CloneableDictionary> dicts = new LinkedHashMap<>();
		
		dicts.put("Java Cloning:", new Dictionary("german", SIZE));
		dicts.put("Copy-Constructor:", new DictionaryConstructor("german", SIZE));
		dicts.put("Reflection:", new DictionaryReflection("german", SIZE));
		dicts.put("Cloner:", new DictionaryCloner("german", SIZE));
		dicts.put("Serialization:", new DictionarySerialization("german", SIZE));
		
		double tref = 0;
		
		for(String method : dicts.keySet()) {
			long start = System.currentTimeMillis();
			for (int i = 0; i < NOFCLONES; i++) {
				dicts.get(method).clone();
			}
			long end = System.currentTimeMillis();
			double t = (end - start) / 1000.0;
			if(tref == 0) tref = t;
			System.out.printf("%-18s%8.3f / %5.2f\n", method, t, t/tref);
		}
	}

}
