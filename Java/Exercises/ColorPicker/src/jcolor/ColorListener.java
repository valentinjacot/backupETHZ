package jcolor;
import java.awt.Color;
@FunctionalInterface
public interface ColorListener {
	void colorValueChanged (Color c);
}