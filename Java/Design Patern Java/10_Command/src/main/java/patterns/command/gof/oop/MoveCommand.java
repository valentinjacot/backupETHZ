package patterns.command.gof.oop;

import patterns.command.gof.Command;
import patterns.command.gof.Figure;

public class MoveCommand implements Command {
	private final Figure f;
	private final int dx, dy;

	public MoveCommand(Figure f, int dx, int dy) {
		this.f = f;
		this.dx = dx;
		this.dy = dy;
	}

	@Override
	public void execute() {
		f.move(dx, dy);
	}
}
