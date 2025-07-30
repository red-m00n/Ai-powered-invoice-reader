import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Download, Edit, Trash2, Search, FileSpreadsheet, FileJson } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import Navbar from "@/components/Navbar";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogClose } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";

// Mock data - replace with real data from your backend
const mockInvoices = [
  {
    id: 1,
    filename: "invoice_001.pdf",
    invoiceNumber: "INV-2024-001",
    invoiceDate: "2024-01-15",
    supplierName: "Tech Solutions Ltd",
    clientName: "ABC Company",
    totalHT: 1000.00,
    tva: 200.00,
    totalTTC: 1200.00,
    createdAt: "2024-01-15 14:30:22"
  },
  {
    id: 2,
    filename: "invoice_002.pdf",
    invoiceNumber: "INV-2024-002",
    invoiceDate: "2024-01-16",
    supplierName: "Digital Services Inc",
    clientName: "XYZ Corp",
    totalHT: 2500.00,
    tva: 500.00,
    totalTTC: 3000.00,
    createdAt: "2024-01-16 10:15:45"
  },
  {
    id: 3,
    filename: "invoice_003.pdf",
    invoiceNumber: "INV-2024-003",
    invoiceDate: "2024-01-17",
    supplierName: "Web Development Co",
    clientName: "StartupXYZ",
    totalHT: 1500.00,
    tva: 300.00,
    totalTTC: 1800.00,
    createdAt: "2024-01-17 16:45:12"
  }
];

const Dashboard = () => {
  const [invoices, setInvoices] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const { toast } = useToast();
  const [editingInvoice, setEditingInvoice] = useState<any | null>(null);
  const [editForm, setEditForm] = useState<any>({});
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isEditLoading, setIsEditLoading] = useState(false);

  const openEdit = (invoice: any) => {
    setEditingInvoice(invoice);
    setEditForm({
      filename: invoice.filename,
      invoice_number: invoice.invoiceNumber,
      invoice_date: invoice.invoiceDate,
      supplier_name: invoice.supplierName,
      total_ht: invoice.totalHT,
      tva: invoice.tva,
      total_ttc: invoice.totalTTC,
    });
    setIsEditOpen(true);
  };

  const closeEdit = () => {
    setIsEditOpen(false);
    setEditingInvoice(null);
    setEditForm({});
  };

  const handleEditChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditForm({ ...editForm, [e.target.name]: e.target.value });
  };

  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingInvoice) return;
    setIsEditLoading(true);
    const formData = new FormData();
    Object.entries(editForm).forEach(([key, value]) => formData.append(key, value as string));
    try {
      const response = await fetch(`/edit/${editingInvoice.id}`, {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        setInvoices((prev) =>
          prev.map((inv) =>
            inv.id === editingInvoice.id
              ? {
                  ...inv,
                  filename: editForm.filename,
                  invoiceNumber: editForm.invoice_number,
                  invoiceDate: editForm.invoice_date,
                  supplierName: editForm.supplier_name,
                  totalHT: Number(editForm.total_ht),
                  tva: Number(editForm.tva),
                  totalTTC: Number(editForm.total_ttc),
                }
              : inv
          )
        );
        toast({
          title: "Invoice updated",
          description: "The invoice has been successfully updated.",
        });
        closeEdit();
      } else {
        const data = await response.json();
        toast({
          title: "Edit failed",
          description: data.error || "An error occurred while editing.",
          variant: "destructive",
        });
      }
    } catch (err) {
      toast({
        title: "Edit failed",
        description: (err as Error).message,
        variant: "destructive",
      });
    } finally {
      setIsEditLoading(false);
    }
  };

  useEffect(() => {
    fetch("/invoices")
      .then((res) => res.json())
      .then((data) => {
        // Map backend fields to frontend fields
        setInvoices(
          data.map((inv: any) => ({
            id: inv.id,
            filename: inv.filename,
            invoiceNumber: inv.invoice_number,
            invoiceDate: inv.invoice_date,
            supplierName: inv.supplier_name,
            clientName: inv.client_name || "", // fallback if not present
            totalHT: Number(inv.total_ht),
            tva: Number(inv.tva),
            totalTTC: Number(inv.total_ttc),
            createdAt: inv.created_at,
          }))
        );
      })
      .catch((err) => {
        toast({
          title: "Failed to fetch invoices",
          description: err.message,
          variant: "destructive",
        });
      });
  }, [toast]);

  const filteredInvoices = invoices.filter(invoice =>
    invoice.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
    invoice.invoiceNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
    invoice.supplierName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    invoice.clientName.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleDelete = async (id: number) => {
    try {
      const response = await fetch(`/invoices/${id}`, {
        method: "DELETE",
      });
      if (response.ok) {
        setInvoices((prev) => prev.filter((invoice) => invoice.id !== id));
        toast({
          title: "Invoice deleted",
          description: "The invoice has been successfully deleted.",
        });
      } else {
        const data = await response.json();
        toast({
          title: "Delete failed",
          description: data.error || "An error occurred while deleting.",
          variant: "destructive",
        });
      }
    } catch (err) {
      toast({
        title: "Delete failed",
        description: (err as Error).message,
        variant: "destructive",
      });
    }
  };

  const handleEdit = (id: number) => {
    const invoice = invoices.find((inv: any) => inv.id === id);
    if (invoice) openEdit(invoice);
  };

  const exportToExcel = () => {
    // In a real app, you'd use a library like xlsx
    toast({
      title: "Export to Excel",
      description: "Excel file would be downloaded here.",
    });
  };

  const exportToJson = () => {
    const dataStr = JSON.stringify(filteredInvoices, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'invoices.json';
    link.click();
    URL.revokeObjectURL(url);
    
    toast({
      title: "Export successful",
      description: "JSON file has been downloaded.",
    });
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'EUR'
    }).format(amount);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
      <Navbar />
      <div className="container mx-auto px-6 py-12">
        <div className="space-y-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 animate-fade-in">
            <div>
              <h1 className="text-4xl font-display font-bold bg-gradient-to-r from-primary via-primary-variant to-secondary bg-clip-text text-transparent">
                Invoice Dashboard
              </h1>
              <p className="text-muted-foreground text-lg mt-2">Manage and view all processed invoices</p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-2">
              <Button onClick={exportToExcel} variant="outline" size="lg" className="hover:bg-primary/10 hover:text-primary border-primary/20">
                <FileSpreadsheet className="w-5 h-5 mr-2" />
                Export Excel
              </Button>
              <Button onClick={exportToJson} variant="outline" size="lg" className="hover:bg-primary/10 hover:text-primary border-primary/20">
                <FileJson className="w-5 h-5 mr-2" />
                Export JSON
              </Button>
            </div>
          </div>

          <div className="relative animate-scale-in">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-secondary/10 rounded-3xl blur-xl"></div>
            <Card className="relative border-0 gradient-card shadow-elegant">
              <CardHeader className="pb-8">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <CardTitle className="text-2xl font-display">All Invoices ({filteredInvoices.length})</CardTitle>
                  <div className="relative max-w-sm">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
                    <Input
                      placeholder="Search invoices..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-12 h-12 border-2 border-primary/20 focus:border-primary/50 bg-background/50"
                    />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border/50 hover:bg-transparent">
                        <TableHead className="font-semibold text-foreground">Filename</TableHead>
                        <TableHead className="font-semibold text-foreground">Invoice Number</TableHead>
                        <TableHead className="font-semibold text-foreground">Invoice Date</TableHead>
                        <TableHead className="font-semibold text-foreground">Supplier Name</TableHead>
                        <TableHead className="font-semibold text-foreground">Client Name</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">Total HT</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">TVA</TableHead>
                        <TableHead className="text-right font-semibold text-foreground">Total TTC</TableHead>
                        <TableHead className="font-semibold text-foreground">Created At</TableHead>
                        <TableHead className="text-center font-semibold text-foreground">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredInvoices.map((invoice) => (
                        <TableRow key={invoice.id} className="border-border/30 hover:bg-primary/5 transition-colors">
                          <TableCell className="font-medium">{invoice.filename}</TableCell>
                          <TableCell>
                          <Badge variant="outline">{invoice.invoiceNumber}</Badge>
                        </TableCell>
                        <TableCell>{invoice.invoiceDate}</TableCell>
                        <TableCell>{invoice.supplierName}</TableCell>
                        <TableCell>{invoice.clientName}</TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(invoice.totalHT)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(invoice.tva)}
                        </TableCell>
                        <TableCell className="text-right font-mono font-semibold">
                          {formatCurrency(invoice.totalTTC)}
                        </TableCell>
                        <TableCell className="text-muted-foreground">
                          {invoice.createdAt}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center justify-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleEdit(invoice.id)}
                            >
                              <Edit className="w-4 h-4" />
                            </Button>
                            
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button variant="ghost" size="sm">
                                  <Trash2 className="w-4 h-4 text-destructive" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Delete Invoice</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    Are you sure you want to delete this invoice? This action cannot be undone.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction
                                    onClick={() => handleDelete(invoice.id)}
                                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                  >
                                    Delete
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                
                {filteredInvoices.length === 0 && (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">No invoices found matching your search.</p>
                  </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      {/* Edit Invoice Modal */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Invoice</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleEditSubmit} className="space-y-4">
            <div>
              <Label>Filename</Label>
              <Input name="filename" value={editForm.filename || ""} onChange={handleEditChange} required />
            </div>
            <div>
              <Label>Invoice Number</Label>
              <Input name="invoice_number" value={editForm.invoice_number || ""} onChange={handleEditChange} />
            </div>
            <div>
              <Label>Invoice Date</Label>
              <Input name="invoice_date" value={editForm.invoice_date || ""} onChange={handleEditChange} />
            </div>
            <div>
              <Label>Supplier Name</Label>
              <Input name="supplier_name" value={editForm.supplier_name || ""} onChange={handleEditChange} />
            </div>
            <div>
              <Label>Total HT</Label>
              <Input name="total_ht" value={editForm.total_ht || ""} onChange={handleEditChange} />
            </div>
            <div>
              <Label>TVA</Label>
              <Input name="tva" value={editForm.tva || ""} onChange={handleEditChange} />
            </div>
            <div>
              <Label>Total TTC</Label>
              <Input name="total_ttc" value={editForm.total_ttc || ""} onChange={handleEditChange} />
            </div>
            <DialogFooter>
              <Button type="submit" disabled={isEditLoading}>{isEditLoading ? "Saving..." : "Save"}</Button>
              <DialogClose asChild>
                <Button type="button" variant="outline" onClick={closeEdit}>Cancel</Button>
              </DialogClose>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Dashboard;