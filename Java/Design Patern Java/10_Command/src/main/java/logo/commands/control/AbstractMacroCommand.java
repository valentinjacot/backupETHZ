package logo.commands.control;

import logo.MacroManager;

public abstract class AbstractMacroCommand implements ControlCommand {
	private final MacroManager macroManager;

	public AbstractMacroCommand(MacroManager macroManager) {
		this.macroManager = macroManager;
	}

	public final MacroManager getMacroManager() { 
		return macroManager;
	}
}
