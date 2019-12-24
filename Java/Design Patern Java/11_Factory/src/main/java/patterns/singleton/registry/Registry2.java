package patterns.singleton.registry;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public enum Registry2 implements Registry {
	INSTANCE;

	private Map<String, Object> entries = Collections.synchronizedMap(new HashMap<String, Object>());

	@Override
	public void register(String name, Object value) {
		entries.put(name, value);
	}

	@Override
	public Object lookup(String name) {
		return entries.get(name);
	}
}
