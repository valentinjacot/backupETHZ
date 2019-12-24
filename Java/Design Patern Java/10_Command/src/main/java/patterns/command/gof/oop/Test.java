package patterns.command.gof.oop;

import patterns.command.gof.Figure;
import patterns.command.gof.Macro;
import patterns.command.gof.Model;

public class Test {
	public static void main(String[] args) {
		Model mod = new Model();
		Figure f1 = new Figure();
		Figure f2 = new Figure();
		
		Macro m = new Macro();
		m.record(new AddFigureCommand(mod, f1));
		m.record(new MoveCommand(f1, 5, 5));
		m.record(new AddFigureCommand(mod, f2));
		m.record(new MoveCommand(f2, 10, 0));
		
		m.run();
	}
}
