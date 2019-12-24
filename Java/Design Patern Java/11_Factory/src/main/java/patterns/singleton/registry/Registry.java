package patterns.singleton.registry;

public interface Registry {
	void register(String name, Object value);
	Object lookup(String name);
}
