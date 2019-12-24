package patterns.observer.fx;

import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;

public class TestMemory {

	public static void main(String[] args) throws Exception {
		StringProperty p1 = new SimpleStringProperty("p1");
		StringProperty p2 = new SimpleStringProperty("p2") {
			@Override public void finalize() { System.out.println("finalized"); }
		};

		//p2.bind(p1);
		p2.bindBidirectional(p1);
		p1.set("xxx");
		
		p2 = null;
		
		for (int i = 0; i < 100; i++) {
			p1.set(""+i);
			System.gc();
		}
		
		// stop program (otherwise everything would be garbage collected anyways
		Object x = new Object();
		synchronized(x) { x.wait(); }
	}

}
