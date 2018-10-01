using System;
//using System.IO;
//uing System.Collections.Generic;
//using System.Linq;

namespace Test
{
    static class TestClass
    {
        class Kuma
        {
            public string a;
            public static string b;

        }
        class Neko
        {
            public string a;
            public static Kuma b;

            public void set(String str)
            {
                a = str;
                b.a = str;
                Kuma.b = str;
            }
            public void print()
            {
                Console.WriteLine(a);
                Console.WriteLine(b);
            }
        }

        [STAThread]
        static void Main(string[] args)
        {
            //Kuma kuma = new Kuma();
            //Console.WriteLine(kuma.a);
            //Console.WriteLine(Kuma.b);
            Neko neko = new Neko();
            neko.set("cat");
            neko.print();
        }
    }
}
