package patterns.observer.twins;

import java.util.Observable;
import java.util.Observer;

@SuppressWarnings("deprecation")
public class ObservableModel extends Model {
	private final ObserverHelper<Model> helper = new ObserverHelper<>(this);

	public Observable getObservableHelper() {
		return helper;
	}
	
	public void addObserver(Observer obs) {
		helper.addObserver(obs);
	}

	@Override
	public void setValue(int value) {
		super.setValue(value);
		helper.setChanged();
		helper.notifyObservers();
	}
}
