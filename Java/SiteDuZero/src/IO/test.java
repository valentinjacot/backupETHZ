package IO;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

//Packages � importer afin d'utiliser les objets

public class test {
	public static void main(String[] args) {
		//Nous d�clarons nos objets en dehors du bloc try/catch
		FileInputStream fis = null;
		FileOutputStream fos = null;

		try {
			//On instancie nos objets :
			//fis va lire le fichier et
			//fos va �crire dans le nouveau !
			fis = new FileInputStream(new File("test.txt"));
			fos = new FileOutputStream(new File("test2_smth.txt"));

			//On cr�e un tableau de byte
			//pour indiquer le nombre de bytes
			//lus � chaque tour de boucle
			byte[] buf = new byte[8];

			//On cr�e une variable de type int
			//pour y affecter le r�sultat de la lecture
			//Vaut -1 quand c'est fini
			int n = 0;

			//Tant que l'affectation dans la variable est possible, on boucle
			//Lorsque la lecture du fichier est termin�e, l'affectation n'est plus possible !
			//On sort donc de la boucle
			while((n = fis.read(buf)) >= 0)
			{
				//On �crit dans notre deuxi�me fichier
				//avec l'objet ad�quat
				fos.write(buf);      
				//On affiche ce qu'a lu notre boucle
				//au format byte et au format char
				for(byte bit : buf)
					System.out.print("\t" + bit + "(" + (char)bit + ")");        
				System.out.println("");
			}

			System.out.println("Copie termin�e !");

		} catch (FileNotFoundException e) {
			//Cette exception est lev�e
			//si l'objet FileInputStream ne trouve aucun fichier
			e.printStackTrace();
		} catch (IOException e) {
			//Celle-ci se produit lors d'une erreur
			//d'�criture ou de lecture
			e.printStackTrace();
		} finally{   
			//On ferme nos flux de donn�es dans un bloc finally pour s'assurer
			//que ces instructions seront ex�cut�es dans tous les cas
			//m�me si une exception est lev�e !
			try{
				if(fis != null)
					fis.close();
				if(fos != null)
					fos.close();
			}catch(IOException e){
				e.printStackTrace();
			}
		}
	}  
}
