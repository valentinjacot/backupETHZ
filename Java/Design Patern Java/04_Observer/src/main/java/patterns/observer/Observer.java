package patterns.observer;

@FunctionalInterface
public interface Observer {
	void update(Observable source);
}
