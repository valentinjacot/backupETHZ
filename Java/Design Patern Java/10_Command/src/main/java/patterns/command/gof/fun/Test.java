package patterns.command.gof.fun;

import patterns.command.gof.Figure;
import patterns.command.gof.Macro;
import patterns.command.gof.Model;

public class Test {
	public static void main(String[] args) {
		Model mod = new Model();
		Figure f1 = new Figure();
		Figure f2 = new Figure();
		
		Macro m = new Macro();
		m.record(() -> mod.addFigure(f1));
		m.record(() -> f1.move(5, 5));
		m.record(() -> mod.addFigure(f2));
		m.record(() -> f2.move(10, 0));
		
		m.run();
	}

}
