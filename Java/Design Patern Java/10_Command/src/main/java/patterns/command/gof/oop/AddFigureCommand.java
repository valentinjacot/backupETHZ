package patterns.command.gof.oop;

import patterns.command.gof.Command;
import patterns.command.gof.Figure;
import patterns.command.gof.Model;

public class AddFigureCommand implements Command {
	private final Model m;
	private final Figure f;
	
	public AddFigureCommand(Model m, Figure f) {
		this.m = m;
		this.f = f;
	}

	@Override
	public void execute() {
		m.addFigure(f);
	}
}
