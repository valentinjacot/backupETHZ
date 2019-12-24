package logo;

import logo.commands.Command;
import logo.commands.turtle.TurtleCommand;

public interface MacroManager {
	boolean isRecordingMacro();
	void addCommand(TurtleCommand command);
	void startMacro(String name);
	Command getCommand(String name);
	void endMacro();
}

