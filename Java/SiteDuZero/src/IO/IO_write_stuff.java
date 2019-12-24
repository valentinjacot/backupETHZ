package IO;

//Packages � importer afin d'utiliser l'objet File
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class IO_write_stuff {
public static void main(String[] args) {
  //Nous d�clarons nos objets en dehors du bloc try/catch
  DataInputStream dis;
  DataOutputStream dos;
  try {
    dos = new DataOutputStream(
            new BufferedOutputStream(
              new FileOutputStream(
                new File("sdz.txt"))));

    //Nous allons �crire chaque type primitif
    dos.writeBoolean(true);
    dos.writeByte(100);
    dos.writeChar('C');
    dos.writeDouble(12.05);
    dos.writeFloat(100.52f);
    dos.writeInt(1024);
    dos.writeLong(123456789654321L);
    dos.writeShort(2);
    dos.close();
      	
    //On r�cup�re maintenant les donn�es !
    dis = new DataInputStream(
            new BufferedInputStream(
              new FileInputStream(
                new File("sdz.txt"))));
          
    System.out.println(dis.readBoolean());
    System.out.println(dis.readByte());
    System.out.println(dis.readChar());
    System.out.println(dis.readDouble());
    System.out.println(dis.readFloat());
    System.out.println(dis.readInt());
    System.out.println(dis.readLong());
    System.out.println(dis.readShort());
      	
  } catch (FileNotFoundException e) {
    e.printStackTrace();
  } catch (IOException e) {
    e.printStackTrace();
  }     	
}
}