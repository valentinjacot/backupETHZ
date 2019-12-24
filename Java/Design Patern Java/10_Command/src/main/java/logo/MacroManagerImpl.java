package logo;

import java.awt.Color;
import java.util.HashMap;
import java.util.Map;

import logo.LogoInterpreter;
import logo.MacroManager;
import logo.commands.Command;
import logo.commands.control.NotFoundCommand;
import logo.commands.turtle.CompositeCommand;
import logo.commands.turtle.TurtleCommand;

public class MacroManagerImpl implements MacroManager {
	private final LogoInterpreter interpreter;

	public MacroManagerImpl(LogoInterpreter interpreter) {
		this.interpreter = interpreter;
	}

	private Map<String, CompositeCommand> macroMap = new HashMap<>();
	private CompositeCommand currentMacro = null;

	@Override
	public boolean isRecordingMacro() {
		return currentMacro != null;
	}

	@Override
	public void addCommand(TurtleCommand command) {
		currentMacro.add(command);
		command.execute();
	}

	@Override
	public void startMacro(String name) {
		if(currentMacro != null) throw new IllegalStateException();
		currentMacro = new CompositeCommand(name);
		interpreter.setColor(Color.RED);
	}
	
	@Override
	public void endMacro() {
		macroMap.put(currentMacro.getName(), currentMacro);
		currentMacro = null;
		interpreter.repaint();
	}


	@Override
	public Command getCommand(String name) {
		if (!macroMap.containsKey(name))
			return new NotFoundCommand("Macro " + name);
		return macroMap.get(name);
	}
}