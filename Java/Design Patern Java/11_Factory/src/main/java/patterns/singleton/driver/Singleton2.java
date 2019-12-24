package patterns.singleton.driver;

import java.io.File;

public enum Singleton2 implements Driver {
	INSTANCE;
	
	@Override
	public String toString(){
		return "singleton";
	}

	@Override
	public void playSong(File file) {
		System.out.println("Yesterday, all my troubles seemed so far away....");
	}

}

