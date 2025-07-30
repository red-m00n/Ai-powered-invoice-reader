import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useToast } from "@/hooks/use-toast";
import { UserPlus, Eye, EyeOff } from "lucide-react";
import Navbar from "@/components/Navbar";

const Users = () => {
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: "",
    role: "user"
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleRoleChange = (value: string) => {
    setFormData({
      ...formData,
      role: value
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (formData.password !== formData.confirmPassword) {
      toast({
        title: "Password mismatch",
        description: "Password and confirm password do not match.",
        variant: "destructive",
      });
      return;
    }

    if (formData.password.length < 6) {
      toast({
        title: "Password too short",
        description: "Password must be at least 6 characters long.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));

    toast({
      title: "User created successfully!",
      description: `User ${formData.fullName} has been added to the system.`,
    });

    setFormData({
      fullName: "",
      email: "",
      password: "",
      confirmPassword: "",
      role: "user"
    });
    
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
      <Navbar />
      <div className="container mx-auto px-6 py-12">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12 animate-fade-in">
            <h1 className="text-4xl font-display font-bold mb-4 bg-gradient-to-r from-primary via-primary-variant to-secondary bg-clip-text text-transparent">
              User Management
            </h1>
            <p className="text-muted-foreground text-lg">
              Create new users who can access the OCR invoice system
            </p>
          </div>

          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-secondary/20 rounded-3xl blur-xl transform scale-110"></div>
            <Card className="relative gradient-card shadow-elegant border-0 backdrop-blur-sm hover-lift">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-2xl font-display">
                  <div className="p-3 rounded-xl gradient-primary">
                    <UserPlus className="w-6 h-6 text-white" />
                  </div>
                  Create New User
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-8">
                  <div className="space-y-3">
                    <Label htmlFor="fullName" className="text-base font-medium">Full Name</Label>
                    <Input
                      id="fullName"
                      name="fullName"
                      type="text"
                      placeholder="Enter full name"
                      value={formData.fullName}
                      onChange={handleChange}
                      required
                      className="h-12 border-2 border-primary/20 focus:border-primary/50 bg-background/50"
                    />
                  </div>

                  <div className="space-y-3">
                    <Label htmlFor="email" className="text-base font-medium">Email Address</Label>
                    <Input
                      id="email"
                      name="email"
                      type="email"
                      placeholder="Enter email address"
                      value={formData.email}
                      onChange={handleChange}
                      required
                      className="h-12 border-2 border-primary/20 focus:border-primary/50 bg-background/50"
                    />
                  </div>

                  <div className="space-y-3">
                    <Label htmlFor="password" className="text-base font-medium">Password</Label>
                    <div className="relative">
                      <Input
                        id="password"
                        name="password"
                        type={showPassword ? "text" : "password"}
                        placeholder="Enter password"
                        value={formData.password}
                        onChange={handleChange}
                        required
                        minLength={6}
                        className="h-12 border-2 border-primary/20 focus:border-primary/50 bg-background/50"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                    <p className="text-sm text-muted-foreground">Must be at least 6 characters long</p>
                  </div>

                  <div className="space-y-3">
                    <Label htmlFor="confirmPassword" className="text-base font-medium">Confirm Password</Label>
                    <div className="relative">
                      <Input
                        id="confirmPassword"
                        name="confirmPassword"
                        type={showConfirmPassword ? "text" : "password"}
                        placeholder="Confirm password"
                        value={formData.confirmPassword}
                        onChange={handleChange}
                        required
                        className="h-12 border-2 border-primary/20 focus:border-primary/50 bg-background/50"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      >
                        {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <Label className="text-base font-medium">User Role</Label>
                    <RadioGroup 
                      value={formData.role} 
                      onValueChange={handleRoleChange}
                      className="flex gap-6"
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="user" id="user" />
                        <Label htmlFor="user" className="cursor-pointer">User</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="admin" id="admin" />
                        <Label htmlFor="admin" className="cursor-pointer">Admin</Label>
                      </div>
                    </RadioGroup>
                    <p className="text-sm text-muted-foreground">Select the user's permission level</p>
                  </div>

                  <Button
                    type="submit" 
                    className="w-full gradient-primary shadow-primary hover:shadow-glow transition-all duration-300" 
                    size="lg"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <div className="animate-spin w-5 h-5 border-2 border-current border-t-transparent rounded-full mr-2"></div>
                        Creating User...
                      </>
                    ) : (
                      <>
                        <UserPlus className="w-5 h-5 mr-2" />
                        Create User
                      </>
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Users;