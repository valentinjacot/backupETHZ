package patterns.observer.sensor;

import java.io.InputStreamReader;
import java.util.Scanner;

public class SensorMain {

	public static void main(String[] args) throws Exception {
		Sensor sensor = new Sensor();
		sensor.addObserver(new ConsoleObserver());
		new LimitingObserver(sensor, 100);
		MinMaxObserver minmax = new MinMaxObserver(sensor);
		AverageObserver avg = new AverageObserver(sensor);
		new QuittingObserver(sensor, 4);
		VisualObserver vo = new VisualObserver(sensor, minmax, avg);
		vo.pack();
		vo.setVisible(true);

		try (Scanner r = new Scanner(new InputStreamReader(System.in))) {
			String s = r.nextLine();
			while (s != null && s.length() > 0) {
				try {
					int t = Integer.parseInt(s);
					sensor.setTemperature(t);
				} catch (IllegalArgumentException e) {
					System.out.println("Illegal Format");
				}
				s = r.nextLine();
			}
		}
		System.exit(0); // necessary if Swing-GUI is running
	}

}
